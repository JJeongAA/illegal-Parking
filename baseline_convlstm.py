import os
import glob
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import csv
from geopy.geocoders import Nominatim

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# [설정] 결과 저장 폴더 지정
# ---------------------------------------------------------
SAVE_DIR = 'convlstm_improved_no_bn_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------
# 0. 시드 고정
# ---------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------
# 1. 개선된 데이터셋 클래스 (강화된 Data Augmentation)
# ---------------------------------------------------------
class IllegalParkingDataset(Dataset):
    def __init__(self, data, window_size=14, horizon=1, stride=1, augment=False):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.augment = augment
        self.num_samples = (len(data) - window_size - horizon) // stride + 1

    def __len__(self):
        return max(0, self.num_samples)

    def _augment_data(self, x):
        """강화된 데이터 증강"""
        aug_x = x.copy()
        
        # 1. 가우시안 노이즈 (기존)
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.005, aug_x.shape)
            aug_x = aug_x + noise
        
        # 2. 스케일링 (밝기 조정)
        if random.random() > 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            aug_x = aug_x * scale_factor
        
        # 3. 시간축 드롭아웃 (일부 시점 마스킹)
        if random.random() > 0.7:
            mask_ratio = 0.1
            num_mask = int(len(aug_x) * mask_ratio)
            mask_indices = np.random.choice(len(aug_x), num_mask, replace=False)
            aug_x[mask_indices] = 0
        
        # 4. 공간적 Cutout (일부 영역 마스킹)
        if random.random() > 0.7:
            _, h, w = aug_x.shape[1], aug_x.shape[2], aug_x.shape[3]
            cutout_size = int(min(h, w) * 0.1)
            x_start = np.random.randint(0, w - cutout_size)
            y_start = np.random.randint(0, h - cutout_size)
            aug_x[:, :, y_start:y_start+cutout_size, x_start:x_start+cutout_size] = 0
        
        # 5. Mixup (두 샘플 혼합) - 시퀀스 내에서
        if random.random() > 0.8 and len(aug_x) > 1:
            lam = np.random.beta(0.2, 0.2)
            idx = np.random.randint(0, len(aug_x))
            aug_x = lam * aug_x + (1 - lam) * np.roll(aug_x, idx, axis=0)
        
        # 클리핑
        aug_x = np.clip(aug_x, 0, 1)
        
        return aug_x

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        target_idx = end_idx + self.horizon
        
        # x shape: (Seq, Channels, H, W)
        x = np.array(self.data[start_idx : end_idx], copy=True)
        y = self.data[end_idx : target_idx]

        if self.augment:
            x = self._augment_data(x)

        x = torch.FloatTensor(x)
        
        # 정답(y)은 Channel 0 (불법주정차)만 예측
        y = torch.FloatTensor(y[-1, 0, :, :]) 

        return x, y

# ---------------------------------------------------------
# 2. 개선된 Loss 함수들
# ---------------------------------------------------------
class CombinedLoss(nn.Module):
    """MSE + SSIM + MAE를 결합한 Loss"""
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # SSIM weight
        self.gamma = gamma  # MAE weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
    def ssim_loss(self, pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
        """SSIM Loss 계산"""
        # 간단한 SSIM 구현
        mu_x = nn.functional.avg_pool2d(pred, window_size, 1, padding=window_size//2)
        mu_y = nn.functional.avg_pool2d(target, window_size, 1, padding=window_size//2)
        
        sigma_x = nn.functional.avg_pool2d(pred ** 2, window_size, 1, padding=window_size//2) - mu_x ** 2
        sigma_y = nn.functional.avg_pool2d(target ** 2, window_size, 1, padding=window_size//2) - mu_y ** 2
        sigma_xy = nn.functional.avg_pool2d(pred * target, window_size, 1, padding=window_size//2) - mu_x * mu_y
        
        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim = ssim_n / ssim_d
        
        return 1 - ssim.mean()
    
    def forward(self, pred, target):
        # pred: (B, 1, H, W), target: (B, H, W) -> unsqueeze 필요
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        ssim_loss_val = self.ssim_loss(pred, target)
        
        total_loss = self.alpha * mse_loss + self.beta * ssim_loss_val + self.gamma * mae_loss
        return total_loss, mse_loss, ssim_loss_val, mae_loss

class FocalLoss(nn.Module):
    """불균형 데이터를 위한 Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        
        mse = (pred - target) ** 2
        focal_weight = self.alpha * (target + 1e-6) ** self.gamma
        focal_loss = (focal_weight * mse).mean()
        
        return focal_loss

# ---------------------------------------------------------
# 3. ConvLSTM 모델 (Batch Normalization 제거)
# ---------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels, 
                              out_channels=4 * hidden_channels, 
                              kernel_size=kernel_size, 
                              padding=padding, 
                              bias=True)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i, f, o, g = torch.sigmoid(cc_i), torch.sigmoid(cc_f), torch.sigmoid(cc_o), torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))

class ConvLSTM(nn.Module):
    """Batch Normalization이 제거된 ConvLSTM"""
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        cell_list = []
        # bn_list 제거!
        
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size))
            
        self.cell_list = nn.ModuleList(cell_list)
        # self.bn_list 제거!
        
        self.output_conv = nn.Conv2d(in_channels=hidden_channels, 
                                     out_channels=1, 
                                     kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):
        batch_size, seq_len, channels, height, width = input_tensor.size()
        
        if hidden_state is None: 
            hidden_state = self._init_hidden(batch_size, (height, width))
            
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                if layer_idx == 0: inp = cur_layer_input[:, t, :, :, :]
                else: inp = cur_layer_input[:, t]
                
                h, c = self.cell_list[layer_idx](inp, [h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            
            # Batch Normalization 제거!
            # 이제 layer_output을 그대로 사용
            
            cur_layer_input = layer_output
            
        last_output = cur_layer_input[:, -1, :, :, :]
        prediction = self.output_conv(last_output)
        return prediction, None

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers): 
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

# ---------------------------------------------------------
# 4. 데이터 로드 및 전처리 (기존 유지)
# ---------------------------------------------------------
def load_enforcement_data(file_paths):
    all_data = []
    print(f"📂 단속 데이터 로드 중... ({len(file_paths)}개 파일)")
    for file_path in file_paths:
        try: df = pd.read_csv(file_path, encoding='cp949') 
        except:
            try: df = pd.read_csv(file_path, encoding='utf-8')
            except: continue
        col_map = {'단속일': 'date', '단속시간': 'time', '위도': 'latitude', '경도': 'longitude'}
        df.rename(columns=col_map, inplace=True)
        if {'latitude', 'longitude', 'date'}.issubset(df.columns):
            all_data.append(df)
    if not all_data: raise ValueError("데이터 로드 실패")
    return pd.concat(all_data, ignore_index=True)

def add_coordinates_if_missing(df):
    # 컬럼이 모두 존재하고 결측치가 없으면 그대로 반환
    if 'latitude' in df.columns and 'longitude' in df.columns:
        if df['latitude'].isnull().sum() == 0 and df['longitude'].isnull().sum() == 0:
            return df

    print("🔍 일부 좌표 결측치에 대해 주소 검색(Geocoding)을 시도합니다...")
    geolocator = Nominatim(user_agent="geo_app_parking_v2")
    
    name_col = next((col for col in df.columns if any(x in col for x in ['장소', '명', 'NAME', 'NM'])), None)
    if not name_col: 
        return df
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
        if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
            try:
                location = geolocator.geocode(f"서울 {row[name_col]}")
                if location:
                    df.at[idx, 'latitude'] = location.latitude
                    df.at[idx, 'longitude'] = location.longitude
            except: 
                pass
    
    return df.dropna(subset=['latitude', 'longitude'])

def create_static_grid(file_path, lat_bins, lon_bins, label="Data"):
    print(f"[{label}] '{os.path.basename(file_path)}' 처리 중...")
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            try: 
                df = pd.read_csv(file_path, encoding='cp949')
            except: 
                df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return None

    # 위도/경도 컬럼 찾기 (우선순위: 한글 > 영문)
    lat_col = None
    lon_col = None
    
    # 1순위: 한글 컬럼명
    for col in df.columns:
        if '위도' in col and lat_col is None:
            lat_col = col
        if '경도' in col and lon_col is None:
            lon_col = col
    
    # 2순위: 영문 컬럼명 (한글이 없을 경우)
    if lat_col is None:
        for col in df.columns:
            col_upper = col.upper()
            if any(x in col_upper for x in ['LAT', 'Y_COORD']) and lat_col is None:
                lat_col = col
    
    if lon_col is None:
        for col in df.columns:
            col_upper = col.upper()
            if any(x in col_upper for x in ['LON', 'X_COORD']) and lon_col is None:
                lon_col = col
    
    # 컬럼 확인
    if lat_col is None or lon_col is None:
        print(f"⚠️ {label} 파일에 위도/경도 컬럼이 없습니다.")
        print(f"   사용 가능한 컬럼: {list(df.columns)}")
        return None
    
    print(f"   위도 컬럼: '{lat_col}', 경도 컬럼: '{lon_col}'")
    
    # 위도/경도만 추출하여 새 DataFrame 생성
    df_clean = pd.DataFrame({
        'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
        'longitude': pd.to_numeric(df[lon_col], errors='coerce')
    })
    
    # 결측치 제거
    df_clean = df_clean.dropna()
    
    if len(df_clean) == 0:
        print(f"⚠️ {label} 데이터에 유효한 좌표가 없습니다.")
        return None
    
    # 그리드 생성
    grid, _, _ = np.histogram2d(df_clean['latitude'], df_clean['longitude'], 
                                bins=[lat_bins, lon_bins])
    print(f"✅ {label} 그리드 생성 완료 (Point: {len(df_clean)})")
    return grid

# ---------------------------------------------------------
# 5. 개선된 학습 함수
# ---------------------------------------------------------
def train_model_improved(model, train_loader, val_loader, num_epochs=100, learning_rate=0.0001, 
                        loss_type='combined', patience=15):
    """
    개선된 학습 함수
    - Combined Loss 또는 Focal Loss 선택 가능
    - Early Stopping 추가
    - 더 자세한 로깅
    """
    
    # Loss 선택
    if loss_type == 'combined':
        criterion = CombinedLoss(alpha=0.7, beta=0.2, gamma=0.1)
        use_combined = True
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        use_combined = False
    else:
        criterion = nn.MSELoss()
        use_combined = False
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 개선된 스케줄러: ReduceLROnPlateau 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5, verbose=True, min_lr=1e-6)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    # 로그 파일
    log_path = os.path.join(SAVE_DIR, 'loss_log.csv')
    with open(log_path, 'w', newline='') as f:
        if use_combined:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'train_mse', 'train_ssim', 'train_mae', 'lr'])
        else:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    print(f"\n🚀 학습 시작 (Loss: {loss_type}, NO Batch Normalization, Early Stopping Patience: {patience})...")
    
    for epoch in range(num_epochs):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_mse_sum = 0.0
        train_ssim_sum = 0.0
        train_mae_sum = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                pred, _ = model(batch_x)
                
                if use_combined:
                    loss, mse_loss, ssim_loss, mae_loss = criterion(pred, batch_y)
                    train_mse_sum += mse_loss.item()
                    train_ssim_sum += ssim_loss.item()
                    train_mae_sum += mae_loss.item()
                else:
                    loss = criterion(pred.squeeze(), batch_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= max(1, len(train_loader))
        train_losses.append(train_loss)

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with autocast():
                    pred, _ = model(batch_x)
                    if use_combined:
                        loss, _, _, _ = criterion(pred, batch_y)
                    else:
                        loss = criterion(pred.squeeze(), batch_y)
                val_loss += loss.item()
        
        val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']

        # 로그 저장
        with open(log_path, 'a', newline='') as f:
            if use_combined:
                avg_mse = train_mse_sum / max(1, len(train_loader))
                avg_ssim = train_ssim_sum / max(1, len(train_loader))
                avg_mae = train_mae_sum / max(1, len(train_loader))
                csv.writer(f).writerow([epoch+1, train_loss, val_loss, avg_mse, avg_ssim, avg_mae, current_lr])
            else:
                csv.writer(f).writerow([epoch+1, train_loss, val_loss, current_lr])

        # 출력
        if (epoch+1) % 5 == 0:
            if use_combined:
                print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.6f} (MSE:{avg_mse:.6f}, SSIM:{avg_ssim:.6f}) | Val: {val_loss:.6f} | LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")
        
        # Best 모델 저장 및 Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(SAVE_DIR, 'best_convlstm_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print(f"✓ Best 모델 저장 (Epoch {epoch+1}, Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n⚠️ Early Stopping 발동 (Patience: {patience})")
            break
            
    return train_losses, val_losses

# ---------------------------------------------------------
# 6. 평가 및 시각화 (기존 유지)
# ---------------------------------------------------------
def evaluate_test_set(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            with autocast():
                pred, _ = model(batch_x)
            all_preds.append(pred.float().cpu().numpy())
            all_targets.append(batch_y.float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = mean_squared_error(all_targets.flatten(), all_preds.flatten())
    mae = mean_absolute_error(all_targets.flatten(), all_preds.flatten())
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())
    return mse, mae, r2, all_preds, all_targets

def visualize_predictions(predictions, targets, num_samples=3):
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))
    if len(indices) == 1: axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        pred, target = predictions[idx].squeeze(), targets[idx].squeeze()
        im1 = axes[i, 0].imshow(target, cmap='hot'); axes[i, 0].set_title('Actual'); plt.colorbar(im1, ax=axes[i, 0])
        im2 = axes[i, 1].imshow(pred, cmap='hot'); axes[i, 1].set_title('Predicted'); plt.colorbar(im2, ax=axes[i, 1])
        im3 = axes[i, 2].imshow(np.abs(target-pred), cmap='coolwarm'); axes[i, 2].set_title('Diff'); plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'prediction_visualization.png'), bbox_inches='tight')
    plt.show()
    print(f"✓ 예측 시각화 저장됨: {os.path.join(SAVE_DIR, 'prediction_visualization.png')}")

# ---------------------------------------------------------
# 7. 메인 실행
# ---------------------------------------------------------
def main():
    print(f"Device: {device}")
    print(f"📂 결과 저장 폴더: {SAVE_DIR}")
    print(f"⚠️ Batch Normalization 미사용 버전")
    
    # 1. 파일 찾기
    data_dir = './illegalparking_dataset'
    enforcement_files = []
    if os.path.exists(data_dir):
        files = glob.glob(os.path.join(data_dir, '*.csv')) + glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        enforcement_files = [f for f in files if ('불법주정차' in f or 'illegal' in f.lower()) and '카메라' not in f]
    if not enforcement_files: return print("❌ 단속 데이터 파일 없음")

    camera_file = r"C:\Users\user\Desktop\convlstm\불법주정차카메라위치파일_데이터처리본.csv"

    # 2. 불법주정차 데이터 로드
    try: enforcement_df = load_enforcement_data(enforcement_files)
    except Exception as e: return print(f"데이터 로드 에러: {e}")
    
    print("\n[전처리 진행]...")
    if 'date' in enforcement_df.columns: 
        enforcement_df['date'] = pd.to_datetime(enforcement_df['date'], format='%Y%m%d', errors='coerce')
    
    df = enforcement_df[(enforcement_df['latitude'] > 33) & (enforcement_df['latitude'] < 39) & 
                        (enforcement_df['longitude'] > 124) & (enforcement_df['longitude'] < 132)]
    
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_bins = np.linspace(lat_min, lat_max, 50 + 1)
    lon_bins = np.linspace(lon_min, lon_max, 50 + 1)
    
    df['time_period'] = df['date'].dt.to_period('D')
    time_periods = sorted(df['time_period'].unique())
    
    time_series_data = []
    for period in tqdm(time_periods, desc="Channel 1 (Parking)"):
        period_data = df[df['time_period'] == period]
        grid, _, _ = np.histogram2d(period_data['latitude'], period_data['longitude'], bins=[lat_bins, lon_bins])
        time_series_data.append(grid)
    time_series_data = np.array(time_series_data)

    time_series_data = np.log1p(time_series_data)
    time_series_data = time_series_data[:, 30:, :] 
    
    valid_indices = time_series_data.sum(axis=(1, 2)) > 0
    time_series_data = time_series_data[valid_indices]
    
    scaler = MinMaxScaler()
    T, H, W = time_series_data.shape
    time_series_data = scaler.fit_transform(time_series_data.reshape(T, -1)).reshape(T, H, W)

    # 3. 카메라 데이터
    lat_bins_cropped = lat_bins[30:]
    camera_grid = None
    if os.path.exists(camera_file):
        cam_raw = create_static_grid(camera_file, lat_bins_cropped, lon_bins, label="Camera")
        if cam_raw is not None:
            cam_scaler = MinMaxScaler()
            camera_grid = cam_scaler.fit_transform(cam_raw.flatten().reshape(-1, 1)).reshape(H, W)
    else: print(f"⚠️ 카메라 파일 없음: {camera_file}")

    # 3.5. POI 데이터 추가
    poi_file = r"C:\Users\user\Desktop\convlstm\서울시_120장소_전체.csv"
    poi_grid = None
    if os.path.exists(poi_file):
        poi_raw = create_static_grid(poi_file, lat_bins_cropped, lon_bins, label="POI")
        if poi_raw is not None:
            poi_scaler = MinMaxScaler()
            poi_grid = poi_scaler.fit_transform(poi_raw.flatten().reshape(-1, 1)).reshape(H, W)
    else: 
        print(f"⚠️ POI 파일 없음: {poi_file}")

    # 4. 2채널 병합
    final_data_list = [np.expand_dims(time_series_data, axis=1)]
    channel_names = ["Parking"]
    
    if camera_grid is not None:
        cam_expanded = np.tile(camera_grid[np.newaxis, np.newaxis, :, :], (len(time_series_data), 1, 1, 1))
        final_data_list.append(cam_expanded)
        channel_names.append("Camera")
    
    if poi_grid is not None:
        poi_expanded = np.tile(poi_grid[np.newaxis, np.newaxis, :, :], (len(time_series_data), 1, 1, 1))
        final_data_list.append(poi_expanded)
        channel_names.append("POI")
        
    final_data = np.concatenate(final_data_list, axis=1)
    input_ch = final_data.shape[1]
    
    print(f"\n✅ 최종 데이터 준비 완료")
    print(f" - Shape: {final_data.shape}")
    print(f" - Channels ({input_ch}): {channel_names}")

    # 5. 분할 및 로더
    test_size = int(len(final_data) * 0.1)
    val_size = int(len(final_data) * 0.1)
    train_size = len(final_data) - val_size - test_size
    
    train_data = final_data[:train_size]
    val_data = final_data[train_size:train_size+val_size]
    test_data = final_data[train_size+val_size:]
    
    train_dataset = IllegalParkingDataset(train_data, augment=True)
    val_dataset = IllegalParkingDataset(val_data, augment=False)
    test_dataset = IllegalParkingDataset(test_data, augment=False)
    
    # Windows에서는 num_workers=0으로 설정 (multiprocessing 에러 방지)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # 6. 모델 생성 및 학습
    print(f"\n모델 생성: ConvLSTM (NO Batch Normalization)")
    model = ConvLSTM(input_channels=input_ch, hidden_channels=128, kernel_size=3, num_layers=3).to(device)
    
    # Combined Loss로 학습 (loss_type='combined', 'focal', 'mse' 중 선택)
    train_losses, val_losses = train_model_improved(
        model, train_loader, val_loader, 
        num_epochs=100, 
        learning_rate=0.0001,
        loss_type='combined',  # 'combined', 'focal', 'mse'
        patience=15
    )
    
    # 7. 평가
    load_path = os.path.join(SAVE_DIR, 'best_convlstm_model.pth')
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ 최고 모델 로드 (Val Loss: {checkpoint['val_loss']:.6f})")
    
    mse, mae, r2, preds, targets = evaluate_test_set(model, test_loader)
    print(f"\n{'='*60}")
    print(f"최종 테스트 결과 (NO Batch Normalization)")
    print(f"{'='*60}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE:      {mse:.6f}")
    print(f"MAE:      {mae:.6f}")
    print(f"{'='*60}")
    
    visualize_predictions(preds, targets)
    
    # 학습 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.title('Training History (NO BN)', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"\n🎉 모든 과정 완료! 결과는 '{SAVE_DIR}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
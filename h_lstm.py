# 해운대 전이학습 LSTM - 3채널 버전 (Parking + CCTV + POI) + pyproj 좌표변환
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
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import csv
from pyproj import Transformer

warnings.filterwarnings('ignore')

# 설정
SEOUL_MODEL_PATH = './lstm_3channel_results/best_lstm_model.pth'
SAVE_DIR = 'haeundae_lstm_transfer_poi_results'
os.makedirs(SAVE_DIR, exist_ok=True)

HAEUNDAE_DATA_DIR = r"C:\Users\user\Desktop\convlstm\부산광역시 해운대구_주정차단속 현황_20231130"
HAEUNDAE_CCTV_FILE = r"C:\Users\user\Desktop\convlstm\haeundae_cctv.csv"
HAEUNDAE_POI_FILE = r"C:\Users\user\Desktop\convlstm\해운대구.csv"

# 좌표 변환기 (EPSG:5179 -> WGS84: EPSG:4326)
transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

# ============================================================================
# 위치 매핑
# ============================================================================
LOCATION_MAP = {
    '해운대해변로': (35.1587, 129.1603), '구남로': (35.1625, 129.1598),
    '동백로': (35.1642, 129.1655), '해운대로': (35.1620, 129.1590),
    '마린시티': (35.1594, 129.1784), '센텀': (35.1691, 129.1313),
    '센텀동로': (35.1691, 129.1313), '센텀서로': (35.1695, 129.1300),
    '우동': (35.1642, 129.1655), '좌동': (35.1688, 129.1778),
    '중동': (35.1625, 129.1598), '송정': (35.1784, 129.2015),
    '재송': (35.1889, 129.1881), '반여': (35.1823, 129.1830),
    '반송': (35.1894, 129.1989), '달맞이': (35.1580, 129.1800),
    '청사포': (35.1600, 129.1900), '미포': (35.1550, 129.1700),
    '벡스코': (35.1689, 129.1360), '영화의전당': (35.1710, 129.1270),
    'default': (35.1650, 129.1650)
}

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

# ============================================================================
# POI 데이터 로드 및 좌표 변환 함수
# ============================================================================
def load_and_transform_poi(file_path):
    """POI 데이터 로드 및 EPSG:5179 -> WGS84 변환"""
    print(f"🔍 POI 데이터 로드 및 좌표 변환 중: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except:
        df = pd.read_csv(file_path, encoding='utf-8')
    
    # x, y 컬럼 찾기
    x_col = next((c for c in df.columns if 'x(EPSG:5179)' in c or 'x' in c.lower()), None)
    y_col = next((c for c in df.columns if 'y(EPSG:5179)' in c or 'y' in c.lower()), None)
    
    if not x_col or not y_col:
        raise ValueError(f"POI 파일에 좌표 컬럼(x/y)이 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
    
    print(f"   사용 컬럼: x='{x_col}', y='{y_col}'")
    
    df = df.dropna(subset=[x_col, y_col])
    
    # 좌표 변환 (EPSG:5179 -> WGS84)
    lon, lat = transformer.transform(df[x_col].values, df[y_col].values)
    
    df['longitude'] = lon
    df['latitude'] = lat
    
    print(f"✅ POI 데이터 변환 완료. 총 {len(df)}개 포인트.")
    return df[['latitude', 'longitude']]

# ============================================================================
# CCTV 데이터 로드 함수
# ============================================================================
def load_cctv_data(file_path):
    """CCTV 데이터 로드"""
    print(f"🔍 CCTV 데이터 로드 중: {os.path.basename(file_path)}")
    
    if not os.path.exists(file_path):
        print(f"⚠️ CCTV 파일 없음: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except:
        df = pd.read_csv(file_path, encoding='utf-8')
    
    # 위도/경도 컬럼 찾기
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if '위도' in col or 'lat' in col_lower or 'y' in col_lower:
            lat_col = col
        if '경도' in col or 'lon' in col_lower or 'x' in col_lower:
            lon_col = col
    
    if not lat_col or not lon_col:
        print(f"⚠️ CCTV 파일에 위도/경도 컬럼이 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
        return None
    
    print(f"   사용 컬럼: 위도='{lat_col}', 경도='{lon_col}'")
    
    df_clean = pd.DataFrame({
        'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
        'longitude': pd.to_numeric(df[lon_col], errors='coerce')
    })
    
    df_clean = df_clean.dropna()
    
    print(f"✅ CCTV 데이터 로드 완료. 총 {len(df_clean)}개 포인트.")
    return df_clean

# ============================================================================
# Weighted Spatial Loss
# ============================================================================
class WeightedSpatialLoss(nn.Module):
    def __init__(self, mask, alpha=0.7, beta=0.3):
        super(WeightedSpatialLoss, self).__init__()
        self.mask = torch.FloatTensor(mask).to(device)
        self.mask = self.mask + 0.1
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        if pred.dim() == 3 and target.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        elif pred.dim() == 2:
            pred = pred.unsqueeze(1).unsqueeze(1)
            target = target.unsqueeze(1).unsqueeze(1)
        
        mse_loss = (self.mse(pred, target) * self.mask).mean()
        mae_loss = (self.mae(pred, target) * self.mask).mean()
        
        return self.alpha * mse_loss + self.beta * mae_loss

# ============================================================================
# Dataset
# ============================================================================
class LSTMParkingDataset(Dataset):
    def __init__(self, data, window_size=14, horizon=1, stride=1, augment=False):
        """
        data: (T, H, W, C) 형태의 그리드 데이터
        """
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.augment = augment
        self.num_samples = (len(data) - window_size - horizon) // stride + 1
        
        self.H, self.W, self.C = data.shape[1], data.shape[2], data.shape[3]
        
    def __len__(self):
        return max(0, self.num_samples)

    def _augment_data(self, x):
        """데이터 증강"""
        aug_x = x.copy()
        
        # 가우시안 노이즈
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.005, aug_x.shape)
            aug_x = aug_x + noise
        
        # 스케일링
        if random.random() > 0.5:
            scale_factor = np.random.uniform(0.9, 1.1)
            aug_x = aug_x * scale_factor
        
        # 클리핑
        aug_x = np.clip(aug_x, 0, 1)
        
        return aug_x

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        target_idx = end_idx + self.horizon
        
        # x shape: (Seq, H, W, C)
        x = np.array(self.data[start_idx : end_idx], copy=True)
        y = self.data[end_idx : target_idx]
        
        if self.augment:
            x = self._augment_data(x)
        
        # (Seq, H, W, C) -> (Seq, C, H, W)
        x = torch.FloatTensor(x).permute(0, 3, 1, 2)
        
        # 정답(y)은 Channel 0 (불법주정차)만 예측
        y = torch.FloatTensor(y[-1, :, :, 0])
        
        return x, y

# ============================================================================
# LSTM 모델 (Batch Norm 미사용)
# ============================================================================
class LSTMModel(nn.Module):
    """
    LSTM 기반 모델 (3채널 지원, Batch Norm 미사용)
    - 입력: (Batch, Seq, Channels, H, W)
    - Flatten → LSTM → Reshape → Conv
    """
    def __init__(self, input_channels, hidden_size, num_layers, output_height, output_width, 
                 dropout=0.2):
        super(LSTMModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_height = output_height
        self.output_width = output_width
        
        # 공간 특징 추출을 위한 Conv 레이어
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # LSTM 입력 차원 계산
        self.lstm_input_dim = 32 * output_height * output_width
        
        # LSTM 레이어들
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 출력 레이어
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_height * output_width)
        )
        
        # 최종 공간 정제를 위한 Conv
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # x: (B, Seq, C, H, W)
        batch_size, seq_len, channels, height, width = x.size()
        
        # 각 시점의 공간 특징 추출
        spatial_features = []
        for t in range(seq_len):
            # (B, C, H, W) → (B, 32, H, W)
            feat = self.spatial_encoder(x[:, t, :, :, :])
            # (B, 32, H, W) → (B, 32*H*W)
            feat_flat = feat.reshape(batch_size, -1)  # view → reshape 변경
            spatial_features.append(feat_flat)
        
        # (B, Seq, 32*H*W)
        lstm_input = torch.stack(spatial_features, dim=1)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # 마지막 시점의 hidden state 사용
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size)
        
        # Decoder: hidden → 공간 그리드
        decoded = self.decoder(last_hidden)  # (B, H*W)
        
        # Reshape to spatial grid
        spatial_output = decoded.reshape(batch_size, 1, self.output_height, self.output_width)  # view → reshape 변경
        
        # 공간 정제
        final_output = self.spatial_decoder(spatial_output)
        
        return final_output.squeeze(1)  # (B, H, W)

# ============================================================================
# Helper Functions
# ============================================================================
def map_location_to_coords(location_str):
    location_str = str(location_str).strip()
    if location_str in LOCATION_MAP: 
        return LOCATION_MAP[location_str]
    for key in sorted(LOCATION_MAP.keys(), key=len, reverse=True):
        if key in location_str and key != 'default': 
            return LOCATION_MAP[key]
    return LOCATION_MAP['default']

def load_haeundae_data():
    csv_files = glob.glob(os.path.join(HAEUNDAE_DATA_DIR, "*.csv"))
    if not csv_files: 
        raise FileNotFoundError("CSV 파일 없음")
    
    df_list = []
    for f in csv_files:
        try: 
            df_list.append(pd.read_csv(f, encoding='cp949'))
        except: 
            df_list.append(pd.read_csv(f, encoding='utf-8', errors='ignore'))
            
    combined_df = pd.concat(df_list, ignore_index=True)
    
    loc_col = next((c for c in combined_df.columns if '장소' in c or '위치' in c or '주소' in c), None)
    date_col = next((c for c in combined_df.columns if '일자' in c or '시간' in c), None)
    
    coords = combined_df[loc_col].apply(map_location_to_coords)
    combined_df['latitude'] = coords.apply(lambda x: x[0]) + np.random.uniform(-0.001, 0.001, len(combined_df))
    combined_df['longitude'] = coords.apply(lambda x: x[1]) + np.random.uniform(-0.001, 0.001, len(combined_df))
    
    combined_df['date'] = pd.to_datetime(combined_df[date_col], errors='coerce')
    combined_df = combined_df.dropna(subset=['date', 'latitude', 'longitude'])
    return combined_df

# ============================================================================
# 학습 함수
# ============================================================================
def train_model(model, train_loader, val_loader, mask, epochs, lr, save_name):
    criterion = WeightedSpatialLoss(mask=mask).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr, weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 로그 파일
    log_path = os.path.join(SAVE_DIR, f'{save_name}_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 로그 저장
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_train, avg_val, current_lr])
        
        print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train:.6f}, Val: {avg_val:.6f}, LR: {current_lr:.2e}")
        
        # Best model 저장
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss
            }, os.path.join(SAVE_DIR, save_name))
            print(f"  ✓ Best model saved (Val Loss: {avg_val:.6f})")
    
    return train_losses, val_losses

# ============================================================================
# 메인 함수
# ============================================================================
def main():
    print(f"Device: {device}")
    print(f"📂 결과 저장 폴더: {SAVE_DIR}")
    print("[1] 데이터 로딩 및 전처리 (3채널: Parking + CCTV + POI)")
    
    # 1. 단속 데이터 로드
    df = load_haeundae_data()
    print(f" - 총 단속 데이터: {len(df):,}건")
    
    # 2. 그리드 설정
    GRID_SIZE = 64
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_bins = np.linspace(lat_min, lat_max, GRID_SIZE + 1)
    lon_bins = np.linspace(lon_min, lon_max, GRID_SIZE + 1)
    
    df['day'] = df['date'].dt.to_period('D')
    periods = sorted(df['day'].unique())
    print(f" - 전체 기간: {len(periods)}일")
    
    # 3. Parking 데이터 생성 (시계열)
    data_frames = []
    
    for p in tqdm(periods, desc="그리드 생성 (Parking)"):
        daily_df = df[df['day'] == p]
        
        # Parking (Gaussian Smoothing)
        grid, _, _ = np.histogram2d(
            daily_df['latitude'], 
            daily_df['longitude'], 
            bins=[lat_bins, lon_bins]
        )
        grid = np.log1p(grid)
        grid = gaussian_filter(grid, sigma=1.0)
        
        if grid.max() > 0: 
            grid = grid / grid.max()
        
        data_frames.append(grid)
    
    parking_data = np.array(data_frames)  # (T, H, W)
    T, H, W = parking_data.shape
    print(f" - Parking Data Shape: {parking_data.shape}")
    
    # 4. CCTV 데이터 로드 (Channel 2)
    cctv_df = load_cctv_data(HAEUNDAE_CCTV_FILE)
    if cctv_df is not None:
        cctv_raw_grid, _, _ = np.histogram2d(
            cctv_df['latitude'], 
            cctv_df['longitude'], 
            bins=[lat_bins, lon_bins]
        )
        cctv_grid = cctv_raw_grid / np.max(cctv_raw_grid) if np.max(cctv_raw_grid) > 0 else np.zeros((H, W))
    else:
        print("⚠️ CCTV 데이터 없음 - 제로 그리드 사용")
        cctv_grid = np.zeros((H, W))
    
    # 5. POI 데이터 로드 (Channel 3)
    poi_df = load_and_transform_poi(HAEUNDAE_POI_FILE)
    if poi_df is not None:
        poi_raw_grid, _, _ = np.histogram2d(
            poi_df['latitude'], 
            poi_df['longitude'], 
            bins=[lat_bins, lon_bins]
        )
        poi_grid = poi_raw_grid / np.max(poi_raw_grid) if np.max(poi_raw_grid) > 0 else np.zeros((H, W))
    else:
        print("⚠️ POI 데이터 없음 - 제로 그리드 사용")
        poi_grid = np.zeros((H, W))
    
    # 6. 3채널 병합 (T, H, W, 3)
    data = np.zeros((T, H, W, 3))
    data[:, :, :, 0] = parking_data  # Channel 0: Parking
    
    # CCTV와 POI는 정적이므로 모든 시점에 동일하게 적용
    for t in range(T):
        data[t, :, :, 1] = cctv_grid   # Channel 1: CCTV
        data[t, :, :, 2] = poi_grid    # Channel 2: POI
    
    print(f"\n✅ 최종 데이터 Shape: {data.shape}")
    print(f" - Channels: [Parking, CCTV, POI]")
    
    # 7. 마스크 생성
    total_parking = data[:, :, :, 0].sum(axis=0)
    spatial_mask = (total_parking > 0).astype(np.float32)
    
    # 8. 데이터셋 분할 (7:2:1)
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    
    train_set = LSTMParkingDataset(data[:train_size], augment=True)
    val_set = LSTMParkingDataset(data[train_size:train_size+val_size])
    test_set = LSTMParkingDataset(data[train_size+val_size:])
    
    print(f" - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, num_workers=0)
    
    # 9. 모델 초기화
    print("\n[2] LSTM 모델 초기화 (3채널, Batch Norm 미사용)")
    model = LSTMModel(
        input_channels=3,
        hidden_size=256,
        num_layers=3,
        output_height=H,
        output_width=W,
        dropout=0.2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" - Total Parameters: {total_params:,}")
    
    # 10. 전이학습: 서울 모델 가중치 로드
    if os.path.exists(SEOUL_MODEL_PATH):
        print(f"\n[3] 서울 모델 가중치 로드: {SEOUL_MODEL_PATH}")
        try:
            checkpoint = torch.load(SEOUL_MODEL_PATH, map_location=device)
            seoul_state_dict = checkpoint['model_state_dict']
            haeundae_state_dict = model.state_dict()
            
            loaded_keys = []
            skipped_keys = []
            
            for name, param in haeundae_state_dict.items():
                if name in seoul_state_dict:
                    seoul_param = seoul_state_dict[name]
                    
                    if param.shape == seoul_param.shape:
                        haeundae_state_dict[name] = seoul_param
                        loaded_keys.append(name)
                    else:
                        skipped_keys.append(f"{name} (shape mismatch)")
                else:
                    skipped_keys.append(f"{name} (not in seoul model)")
            
            model.load_state_dict(haeundae_state_dict)
            print(f" ✅ 전이학습 완료:")
            print(f"    - 로드된 레이어: {len(loaded_keys)}개")
            print(f"    - 스킵된 레이어: {len(skipped_keys)}개")
            
        except Exception as e:
            print(f"⚠️ 가중치 로드 실패: {e}")
            print("   랜덤 초기화 상태로 시작합니다.")
    else:
        print(f"⚠️ 서울 모델 없음: {SEOUL_MODEL_PATH}")
        print("   랜덤 초기화 상태로 시작합니다.")
    
    # 11. 학습
    all_train_losses = []
    all_val_losses = []
    
    # STEP 1: Head Training
    print("\n[4] 학습 시작 - STEP 1: Head Training (Decoder만 학습)")
    for name, p in model.named_parameters():
        if "decoder" not in name and "spatial_decoder" not in name:
            p.requires_grad = False
    
    train_losses_1, val_losses_1 = train_model(
        model, train_loader, val_loader, spatial_mask,
        epochs=10, lr=1e-3, save_name='stage1.pth'
    )
    all_train_losses.extend(train_losses_1)
    all_val_losses.extend(val_losses_1)
    
    # STEP 2: Fine Tuning
    print("\n[5] 학습 시작 - STEP 2: Fine Tuning (전체 학습)")
    for p in model.parameters():
        p.requires_grad = True
    
    train_losses_2, val_losses_2 = train_model(
        model, train_loader, val_loader, spatial_mask,
        epochs=40, lr=1e-4, save_name='best_haeundae_lstm.pth'
    )
    all_train_losses.extend(train_losses_2)
    all_val_losses.extend(val_losses_2)
    
    # 12. Training History 시각화
    print("\n[6] Training History 저장 중...")
    plt.figure(figsize=(10, 5))
    plt.plot(all_train_losses, label='Train Loss', linewidth=2)
    plt.plot(all_val_losses, label='Val Loss', linewidth=2)
    plt.axvline(x=10, color='red', linestyle='--', label='Fine-tuning Start', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM 3-Channel Training History (Transfer Learning with POI)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=150)
    plt.close()
    print(f" ✓ 학습 곡선 저장: training_history.png")
    
    # 13. 최종 평가
    print("\n[7] 최종 평가")
    if os.path.exists(os.path.join(SAVE_DIR, 'best_haeundae_lstm.pth')):
        checkpoint = torch.load(os.path.join(SAVE_DIR, 'best_haeundae_lstm.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.numpy())
    
    spatial_preds = np.concatenate(preds_list, axis=0)
    spatial_targets = np.concatenate(targets_list, axis=0)
    
    flat_preds = spatial_preds.flatten()
    flat_targets = spatial_targets.flatten()
    
    # 메트릭
    r2 = r2_score(flat_targets, flat_preds)
    mse = mean_squared_error(flat_targets, flat_preds)
    mae = mean_absolute_error(flat_targets, flat_preds)
    
    print(f"\n{'='*60}")
    print(f"최종 성능 평가 (LSTM 3-Channel Transfer Learning + POI)")
    print(f"{'='*60}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE:      {mse:.6f}")
    print(f"MAE:      {mae:.6f}")
    print(f"{'='*60}\n")
    
    # 14. 시각화 1: Line Plot
    positive_indices = np.where(flat_targets > 0.01)[0]
    if len(positive_indices) > 0:
        sample_limit = min(100, len(positive_indices))
        sample_idx = positive_indices[:sample_limit]
        
        plt.figure(figsize=(12, 5))
        plt.plot(flat_targets[sample_idx], label='Actual', marker='o', markersize=3)
        plt.plot(flat_preds[sample_idx], label='Predicted', marker='x', linestyle='--', markersize=3)
        plt.title(f"LSTM 3-Ch Prediction (R²={r2:.3f}, MAE={mae:.4f})")
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Parking Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(SAVE_DIR, 'prediction_line.png'), dpi=150)
        plt.close()
        print(" ✓ 라인 차트 저장: prediction_line.png")
    
    # 15. 시각화 2: Heatmap
    image_sums = spatial_targets.sum(axis=(1, 2))
    top_indices = np.argsort(image_sums)[-3:][::-1]
    
    if image_sums.max() == 0:
        top_indices = np.random.choice(len(spatial_targets), 3, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, idx in enumerate(top_indices):
        target_img = spatial_targets[idx]
        pred_img = spatial_preds[idx]
        
        vmin = 0
        vmax = max(target_img.max(), pred_img.max())
        if vmax == 0: vmax = 1
        
        # Actual
        im1 = axes[i, 0].imshow(target_img, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Actual (Sample {idx})', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        fig.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Predicted
        im2 = axes[i, 1].imshow(pred_img, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('LSTM Predicted', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')
        fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Error
        diff_img = target_img - pred_img
        div_norm = max(abs(diff_img.min()), abs(diff_img.max()))
        if div_norm == 0: div_norm = 1
        
        im3 = axes[i, 2].imshow(diff_img, cmap='coolwarm', vmin=-div_norm, vmax=div_norm)
        sample_mae = np.mean(np.abs(diff_img))
        axes[i, 2].set_title(f'Error (MAE: {sample_mae:.4f})', fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')
        fig.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'LSTM 3-Ch Prediction Heatmap (Test R²={r2:.3f})', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'prediction_heatmap.png'), dpi=150)
    plt.close()
    print(" ✓ 히트맵 저장: prediction_heatmap.png")
    
    print("\n[완료] 모든 프로세스 종료")
    print(f"\n결과 파일 저장 위치: {SAVE_DIR}/")
    print(f"  - training_history.png")
    print(f"  - prediction_line.png")
    print(f"  - prediction_heatmap.png")
    print(f"  - best_haeundae_lstm.pth")

if __name__ == "__main__":
    main()
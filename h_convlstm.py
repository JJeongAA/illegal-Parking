# 해운대 전이학습 - Batch Norm 제거 버전 (3채널: 주정차, CCTV, POI)
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
import re
from pyproj import Transformer

warnings.filterwarnings('ignore')

# 설정
SEOUL_MODEL_PATH = './convlstm_improved_no_bn_results/best_convlstm_model.pth'
SAVE_DIR = 'haeundae_transfer_no_bn_3ch_results'
os.makedirs(SAVE_DIR, exist_ok=True)

HAEUNDAE_DATA_DIR = r"C:\Users\user\Desktop\convlstm\부산광역시 해운대구_주정차단속 현황_20231130"
HAEUNDAE_CCTV_FILE = r"C:\Users\user\Desktop\convlstm\haeundae_cctv.csv"
HAEUNDAE_POI_FILE = r"C:\Users\user\Desktop\convlstm\해운대구.csv"

# 좌표 변환기 (EPSG:5179 -> WGS84: EPSG:4326)
transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

# POI 데이터 로드 및 좌표 변환 함수
def load_and_transform_poi(file_path):
    print(f"🔍 POI 데이터 로드 및 좌표 변환 중: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='cp949')
    except:
        df = pd.read_csv(file_path, encoding='utf-8')
    
    x_col = next((c for c in df.columns if c.startswith('x(EPSG:5179)')), None)
    y_col = next((c for c in df.columns if c.startswith('y(EPSG:5179)')), None)
    
    if not x_col or not y_col:
        raise ValueError("POI 파일에 EPSG:5179 좌표 컬럼(x/y)이 없습니다.")
    
    df = df.dropna(subset=[x_col, y_col])
    lon, lat = transformer.transform(df[x_col].values, df[y_col].values)
    
    df['longitude'] = lon
    df['latitude'] = lat
    
    print(f"✅ POI 데이터 변환 완료. 총 {len(df)}개 포인트.")
    return df[['latitude', 'longitude']]

# 위치 매핑
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

# Weighted Mask Loss
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
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        mse_loss = (self.mse(pred, target) * self.mask).mean()
        mae_loss = (self.mae(pred, target) * self.mask).mean()
        return self.alpha * mse_loss + self.beta * mae_loss

# 데이터셋
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

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        target_idx = end_idx + self.horizon
        x = np.array(self.data[start_idx : end_idx], copy=True)
        y = self.data[end_idx : target_idx]
        if self.augment and random.random() > 0.5:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise
        x = torch.FloatTensor(x).permute(0, 3, 1, 2)
        y = torch.FloatTensor(y[-1, :, :, 0])
        return x, y

# ConvLSTM 모델 (Batch Norm 제거)
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

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
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_channels if i == 0 else hidden_channels
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_channels, kernel_size))
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, input_tensor):
        batch_size, seq_len, _, height, width = input_tensor.size()
        hidden_state = self._init_hidden(batch_size, (height, width))
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                inp = cur_layer_input[:, t]
                h, c = self.cell_list[layer_idx](inp, [h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
        return self.output_conv(cur_layer_input[:, -1]), None

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.cell_list]

# Helper Functions
def map_location_to_coords(location_str):
    location_str = str(location_str).strip()
    if location_str in LOCATION_MAP: return LOCATION_MAP[location_str]
    for key in sorted(LOCATION_MAP.keys(), key=len, reverse=True):
        if key in location_str and key != 'default': return LOCATION_MAP[key]
    return LOCATION_MAP['default']

def load_haeundae_data():
    csv_files = glob.glob(os.path.join(HAEUNDAE_DATA_DIR, "*.csv"))
    if not csv_files: raise FileNotFoundError("CSV 파일 없음")
    df_list = []
    for f in csv_files:
        try: df_list.append(pd.read_csv(f, encoding='cp949'))
        except: df_list.append(pd.read_csv(f, encoding='utf-8', errors='ignore'))
    combined_df = pd.concat(df_list, ignore_index=True)
    loc_col = next((c for c in combined_df.columns if '장소' in c or '위치' in c or '주소' in c), None)
    date_col = next((c for c in combined_df.columns if '일자' in c or '시간' in c), None)
    coords = combined_df[loc_col].apply(map_location_to_coords)
    combined_df['latitude'] = coords.apply(lambda x: x[0]) + np.random.uniform(-0.001, 0.001, len(combined_df))
    combined_df['longitude'] = coords.apply(lambda x: x[1]) + np.random.uniform(-0.001, 0.001, len(combined_df))
    combined_df['date'] = pd.to_datetime(combined_df[date_col], errors='coerce')
    combined_df = combined_df.dropna(subset=['date', 'latitude', 'longitude'])
    return combined_df

# 학습 루프
def train_model(model, train_loader, val_loader, mask, epochs, lr, save_name):
    criterion = WeightedSpatialLoss(mask=mask).to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                pred, _ = model(x)
                loss = criterion(pred.squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    pred, _ = model(x)
                    loss = criterion(pred.squeeze(1), y)
                val_loss += loss.item()
        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train:.6f}, Val: {avg_val:.6f}")
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(SAVE_DIR, save_name))
            print(f"  ✓ Best model saved (Val Loss: {avg_val:.6f})")
    return train_losses, val_losses

def main():
    print(f"Device: {device}")
    print("[1] 데이터 처리 및 그리드 생성 시작 (3채널: 주정차, CCTV, POI)")
    df = load_haeundae_data()
    print(f" - 총 단속 데이터: {len(df):,}건")
    GRID_SIZE = 64
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_bins = np.linspace(lat_min, lat_max, GRID_SIZE + 1)
    lon_bins = np.linspace(lon_min, lon_max, GRID_SIZE + 1)
    df['day'] = df['date'].dt.to_period('D')
    periods = sorted(df['day'].unique())
    print(f" - 전체 기간: {len(periods)}일")
    try:
        poi_df = load_and_transform_poi(HAEUNDAE_POI_FILE)
    except ValueError as e:
        print(f"❌ POI 데이터 로드 실패: {e}")
        return
    poi_raw_grid, _, _ = np.histogram2d(poi_df['latitude'], poi_df['longitude'], bins=[lat_bins, lon_bins])
    poi_grid = poi_raw_grid / np.max(poi_raw_grid) if np.max(poi_raw_grid) > 0 else np.zeros_like(poi_raw_grid)
    H, W = poi_grid.shape
    cctv_grid = np.zeros((H, W))
    data_frames = []
    for p in tqdm(periods, desc="Generating Grids (3 Channels)"):
        daily_df = df[df['day'] == p]
        grid, _, _ = np.histogram2d(daily_df['latitude'], daily_df['longitude'], bins=[lat_bins, lon_bins])
        grid = np.log1p(grid)
        grid = gaussian_filter(grid, sigma=1.0)
        if grid.max() > 0: grid = grid / grid.max()
        combined = np.stack([grid, cctv_grid, poi_grid], axis=-1)
        data_frames.append(combined)
    data = np.array(data_frames)
    INPUT_CHANNELS = 3
    print(f"Data Shape: {data.shape}")
    total_parking = data[:, :, :, 0].sum(axis=0)
    spatial_mask = (total_parking > 0).astype(np.float32)
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    train_set = IllegalParkingDataset(data[:train_size], augment=True)
    val_set = IllegalParkingDataset(data[train_size:train_size+val_size])
    test_set = IllegalParkingDataset(data[train_size+val_size:])
    print(f" - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)
    print("\n[2] 모델 초기화 및 전이학습 설정 (Batch Norm 제거)")
    model = ConvLSTM(input_channels=INPUT_CHANNELS, hidden_channels=128, kernel_size=3, num_layers=3).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" - Total Parameters: {total_params:,}")
    if os.path.exists(SEOUL_MODEL_PATH):
        print(f" - 서울 모델 로드 중: {SEOUL_MODEL_PATH}")
        chk = torch.load(SEOUL_MODEL_PATH, map_location=device)
        seoul_state_dict = chk['model_state_dict']
        haeundae_state_dict = model.state_dict()
        for name, param in haeundae_state_dict.items():
            if 'bn_list' in name:
                continue
            if name in seoul_state_dict:
                seoul_param = seoul_state_dict[name]
                if param.shape == seoul_param.shape:
                    haeundae_state_dict[name] = seoul_param
                elif 'cell_list.0.conv.weight' in name:
                    if seoul_param.shape[1] == 2 + 128:
                        print(f"   -> 입력 채널 확장 적용 (2ch -> 3ch): {name}")
                        new_weight = param.clone()
                        src_input = 2
                        new_weight[:, :src_input, :, :] = seoul_param[:, :src_input, :, :]
                        new_weight[:, src_input+1:, :, :] = seoul_param[:, src_input:, :, :]
                        haeundae_state_dict[name] = new_weight
                    else:
                        print(f"   -> 입력 크기 불일치 또는 3채널 전이: {name} (크기: {seoul_param.shape[1]})")
        model.load_state_dict(haeundae_state_dict, strict=False)
        print(" - 가중치 이식 완료 (Transfer Learning Ready)")
    else:
        print("⚠️ 서울 모델 없음. 랜덤 초기화 상태로 시작.")
    all_train_losses = []
    all_val_losses = []
    print("\n[3] 학습 시작 - STEP 1: Head Training")
    for name, p in model.named_parameters():
        if "output_conv" not in name: p.requires_grad = False
    train_losses_1, val_losses_1 = train_model(model, train_loader, val_loader, spatial_mask, epochs=10, lr=1e-3, save_name='stage1.pth')
    all_train_losses.extend(train_losses_1)
    all_val_losses.extend(val_losses_1)
    print("\n[3] 학습 시작 - STEP 2: Fine Tuning")
    for p in model.parameters(): p.requires_grad = True
    train_losses_2, val_losses_2 = train_model(model, train_loader, val_loader, spatial_mask, epochs=40, lr=1e-4, save_name='best_haeundae.pth')
    all_train_losses.extend(train_losses_2)
    all_val_losses.extend(val_losses_2)
    print("\n[4] Training History 저장 중...")
    plt.figure(figsize=(10, 5))
    plt.plot(all_train_losses, label='Train Loss', linewidth=2)
    plt.plot(all_val_losses, label='Val Loss', linewidth=2)
    plt.axvline(x=10, color='red', linestyle='--', label='Fine-tuning Start', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ConvLSTM Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=150)
    plt.close()
    print(f" - 학습 곡선 저장: {os.path.join(SAVE_DIR, 'training_history.png')}")
    print("\n[5] 최종 평가 및 시각화")
    if os.path.exists(os.path.join(SAVE_DIR, 'best_haeundae.pth')):
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_haeundae.pth'))['model_state_dict'])
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred, _ = model(x)
            preds_list.append(pred.squeeze(1).cpu().numpy())
            targets_list.append(y.numpy())
    spatial_preds = np.concatenate(preds_list, axis=0)
    spatial_targets = np.concatenate(targets_list, axis=0)
    flat_preds = spatial_preds.flatten()
    flat_targets = spatial_targets.flatten()
    r2 = r2_score(flat_targets, flat_preds)
    mse = mean_squared_error(flat_targets, flat_preds)
    mae = mean_absolute_error(flat_targets, flat_preds)
    print(f"\n{'='*50}")
    print(f"최종 성능 평가 (ConvLSTM - No BatchNorm)")
    print(f"{'='*50}")
    print(f"R² Score: {r2:.4f}")
    print(f"MSE:      {mse:.6f}")
    print(f"MAE:      {mae:.6f}")
    print(f"{'='*50}\n")
    positive_indices = np.where(flat_targets > 0.01)[0]
    if len(positive_indices) > 0:
        sample_limit = min(100, len(positive_indices))
        sample_idx = positive_indices[:sample_limit]
        plt.figure(figsize=(12, 5))
        plt.plot(flat_targets[sample_idx], label='Actual', marker='o', markersize=3, linewidth=1.5)
        plt.plot(flat_preds[sample_idx], label='Predicted', marker='x', linestyle='--', markersize=3, linewidth=1.5)
        plt.title(f"ConvLSTM Prediction (R²={r2:.3f}, MAE={mae:.4f})")
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Parking Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'prediction_line.png'), dpi=150)
        plt.close()
        print(f" - 라인 차트 저장: {os.path.join(SAVE_DIR, 'prediction_line.png')}")
    print(" - 히트맵 생성 중...")
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
        im1 = axes[i, 0].imshow(target_img, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Actual (Sample {idx})', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        fig.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        im2 = axes[i, 1].imshow(pred_img, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('ConvLSTM Predicted', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')
        fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        diff_img = target_img - pred_img
        div_norm = max(abs(diff_img.min()), abs(diff_img.max()))
        if div_norm == 0: div_norm = 1
        im3 = axes[i, 2].imshow(diff_img, cmap='coolwarm', vmin=-div_norm, vmax=div_norm)
        sample_mae = np.mean(np.abs(diff_img))
        axes[i, 2].set_title(f'Error (MAE: {sample_mae:.4f})', fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')
        fig.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
    plt.suptitle(f'ConvLSTM Prediction Heatmap (Test R²={r2:.3f})', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'prediction_heatmap.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" - 히트맵 저장 완료: {save_path}")
    print("\n[완료] 모든 프로세스가 정상적으로 종료되었습니다.")
    print(f"\n결과 파일 저장 위치: {SAVE_DIR}/")
    print(f"  - training_history.png")
    print(f"  - prediction_line.png")
    print(f"  - prediction_heatmap.png")
    print(f"  - best_haeundae.pth")

if __name__ == "__main__":
    main()


# =============================================================================
# STEP 1: 필수 패키지 설치
# =============================================================================
!pip install geopandas -q

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import geopandas as gpd
from shapely.geometry import Point
import warnings

warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*70}\n")

# =============================================================================
# STEP 3: 모델 클래스 정의
# =============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(
                input_channels=cur_input_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size
            ))
        self.cell_list = nn.ModuleList(cell_list)
        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1
        )

    def forward(self, input_tensor, hidden_state=None):
        batch_size, seq_len, channels, height, width = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))
        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        last_output = layer_output_list[-1]
        prediction = self.output_conv(last_output[:, -1, :, :, :])
        return prediction, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

class IllegalParkingDataset(Dataset):
    def __init__(self, data, sequence_length=6, prediction_length=1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

    def __len__(self):
        return max(0, len(self.data) - self.sequence_length - self.prediction_length + 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length:idx+self.sequence_length+self.prediction_length]
        x = torch.FloatTensor(x).unsqueeze(1)
        y = torch.FloatTensor(y[-1])
        return x, y

# =============================================================================
# STEP 4: 데이터 처리 함수
# =============================================================================

def load_enforcement_data(file_paths):
    all_data = []
    print("\n" + "="*70)
    print("단속 데이터 로딩")
    print("="*70)
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"\n로딩: {file_path}")
            try:
                df = None
                for encoding in ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"  ✓ 성공 (인코딩: {encoding})")
                        break
                    except:
                        continue
                if df is None:
                    print(f"  ✗ 실패")
                    continue
                
                column_mapping = {
                    '단속일': 'date', '단속시간': 'time', '구주소': 'old_address',
                    '도로명': 'road_name', '경도': 'longitude', '위도': 'latitude'
                }
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df.rename(columns={old_name: new_name}, inplace=True)
                all_data.append(df)
            except Exception as e:
                print(f"  ✗ 에러: {str(e)}")
                continue
    if not all_data:
        raise ValueError("로드된 데이터가 없습니다!")
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def create_spatiotemporal_grid(df, grid_size=(50, 50), time_unit='W'):
    """
    좌표 필터링(한국) 및 SHP 무시하고 데이터 분포 기반 그리드 생성
    """
    print("\n" + "="*70)
    print(f"시공간 그리드 생성 (단위: {time_unit})")
    print("="*70)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

    # [중요] 좌표 이상치 제거 (한국 범위 근사)
    print("✓ 좌표 이상치(0, 0 등) 및 한국 범위 밖 데이터 제거 중...")
    original_len = len(df)
    df = df[
        (df['latitude'] > 33) & (df['latitude'] < 39) &
        (df['longitude'] > 124) & (df['longitude'] < 132)
    ]
    print(f"  - 제거된 이상치 데이터: {original_len - len(df):,}건")
    print(f"  - 유효 데이터: {len(df):,}건")

    # 데이터의 실제 범위 사용
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    print(f"✓ 그리드 범위: 위도({lat_min:.4f}~{lat_max:.4f}), 경도({lon_min:.4f}~{lon_max:.4f})")

    df['time_period'] = df['date'].dt.to_period(time_unit)
    time_periods = sorted(df['time_period'].unique())

    print(f"✓ 기간: {time_periods[0]} ~ {time_periods[-1]} (총 {len(time_periods)}개 Time Step)")

    height, width = grid_size
    lat_bins = np.linspace(lat_min, lat_max, height + 1)
    lon_bins = np.linspace(lon_min, lon_max, width + 1)

    time_series_data = []

    print("✓ 시간대별 그리드 집계 중...")
    for period in tqdm(time_periods):
        period_data = df[df['time_period'] == period]
        grid, _, _ = np.histogram2d(
            period_data['latitude'],
            period_data['longitude'],
            bins=[lat_bins, lon_bins]
        )
        time_series_data.append(grid)

    time_series_data = np.array(time_series_data) # (T, H, W)

    metadata = {
        'time_periods': [str(p) for p in time_periods],
        'lat_range': (lat_min, lat_max),
        'lon_range': (lon_min, lon_max),
        'grid_size': grid_size,
        'lat_bins': lat_bins,
        'lon_bins': lon_bins
    }

    return time_series_data, metadata

# =============================================================================
# STEP 5: 학습 및 평가 함수 (Weight Decay 적용)
# =============================================================================

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    # [중요] weight_decay 추가로 과적합 방지
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("\n학습 시작 (Weight Decay 적용)...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader) if len(train_loader) > 0 else 1
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred, _ = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, 'best_convlstm_model.pth')

    return train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()

def evaluate_test_set(model, test_loader):
    print("\n테스트 세트 평가 중...")
    model.eval()
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    test_loss = 0.0
    test_mae = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred, _ = model(batch_x)
            loss = criterion(pred, batch_y)
            mae = mae_criterion(pred, batch_y)
            
            test_loss += loss.item()
            test_mae += mae.item()
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader) if len(test_loader) > 0 else 1
    test_mae /= len(test_loader) if len(test_loader) > 0 else 1
    
    if all_preds:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        try:
            r2 = r2_score(all_targets.flatten(), all_preds.flatten())
        except:
            r2 = 0.0
    else:
        r2 = 0.0

    return test_loss, test_mae, r2, all_preds, all_targets

def visualize_predictions(predictions, targets, num_samples=3):
    if len(predictions) == 0: return
    num_samples = min(num_samples, len(predictions))
    indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        pred = predictions[idx].squeeze()
        target = targets[idx].squeeze()
        
        im1 = axes[i, 0].imshow(target, cmap='hot')
        axes[i, 0].set_title(f'Actual (Sample {idx})')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(pred, cmap='hot')
        axes[i, 1].set_title(f'Predicted (Sample {idx})')
        plt.colorbar(im2, ax=axes[i, 1])
        
        im3 = axes[i, 2].imshow(np.abs(target-pred), cmap='coolwarm')
        axes[i, 2].set_title(f'Diff (Sample {idx})')
        plt.colorbar(im3, ax=axes[i, 2])
        
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def transform_series(series, scaler):
    if series is None or len(series) == 0: return series
    original_shape = series.shape
    flat = series.reshape(original_shape[0], -1)
    scaled = scaler.transform(flat)
    return scaled.reshape(original_shape)

# =============================================================================
# STEP 6: 메인 실행
# =============================================================================
def main():
    # 1. 파일 로드
    csv_files = glob.glob('./data/*.csv') + glob.glob('./*.csv')
    enforcement_files = [f for f in csv_files if '불법주정차' in f or 'illegal' in f.lower()]
    
    if not enforcement_files:
        print("❌ 단속 데이터 파일이 없습니다.")
        return

    enforcement_df = load_enforcement_data(enforcement_files)

  
    time_series_data, metadata = create_spatiotemporal_grid(
        df=enforcement_df,
        grid_size=(50, 50),
        time_unit='D'  # (M:월별, W:주별, D:일별)
    )

    # 3. 로그 변환
    print("\n[데이터 전처리] 로그 변환 수행 (np.log1p)...")
    time_series_data = np.log1p(time_series_data)
    
    # 빈 데이터(All Zero) 삭제
    valid_indices = time_series_data.sum(axis=(1, 2)) > 0
    time_series_data = time_series_data[valid_indices]
    print(f"⚠️ 빈 데이터(All Zeros) 삭제 후 남은 데이터: {len(time_series_data)}개")
    
    if len(time_series_data) < 100:
        print("🚨 경고: 일 단위로 변환했음에도 데이터가 적습니다. 원본 데이터를 확인해보세요.")
    
    # 4. 데이터 분할 (비율 기반)
    total_len = len(time_series_data)
    
    # 시퀀스 길이 확장
    # 일 단위 데이터이므로, 14일(2주) 패턴을 보고 다음날을 예측하도록 설정
    sequence_length = 14 
    prediction_length = 1
    
    # 데이터가 많아졌으므로 Test/Val 비율 유지
    test_size = int(total_len * 0.1)
    val_size = int(total_len * 0.1)
    train_size = total_len - val_size - test_size

    train_data = time_series_data[:train_size]
    val_data = time_series_data[train_size:train_size+val_size]
    test_data = time_series_data[train_size+val_size:]

    print(f"\n데이터 분할 결과 (일 단위):")
    print(f"  Train({len(train_data)}) / Val({len(val_data)}) / Test({len(test_data)})")

    # 5. 스케일링
    scaler = StandardScaler()
    train_flat = train_data.reshape(len(train_data), -1)
    train_scaled = scaler.fit_transform(train_flat).reshape(train_data.shape)
    
    val_scaled = transform_series(val_data, scaler)
    test_scaled = transform_series(test_data, scaler)

    # 6. 데이터셋 생성 (Batch Size 증가)
    batch_size = 32  # 데이터가 많아졌으므로 배치를 늘려 학습 안정화
    
    
    train_dataset = IllegalParkingDataset(train_scaled, sequence_length=sequence_length)
    val_dataset = IllegalParkingDataset(val_scaled, sequence_length=sequence_length)
    test_dataset = IllegalParkingDataset(test_scaled, sequence_length=sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvLSTM(
        input_channels=1, 
        hidden_channels=128, 
        kernel_size=3, 
        num_layers=1 
    ).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)
    plot_training_history(train_losses, val_losses)

    # 8. 평가
    if os.path.exists('best_convlstm_model.pth'):
        print("\n최고 성능 모델 로드 중...")
        checkpoint = torch.load('best_convlstm_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Best Val Loss: {checkpoint['val_loss']:.6f} (Epoch {checkpoint['epoch']})")
    
    loss, mae, r2, preds, targets = evaluate_test_set(model, test_loader)
    print(f"\n최종 테스트 결과: MSE={loss:.6f}, MAE={mae:.6f}, R2={r2:.4f}")
    
    visualize_predictions(preds, targets)
    print("\n🎉 완료되었습니다.")

if __name__ == "__main__":
    main()
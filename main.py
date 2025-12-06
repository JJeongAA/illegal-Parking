from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Optional
import os

# Hugging Face Hub
from huggingface_hub import hf_hub_download

app = FastAPI(title="불법주정차 예측 시스템")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}

# ✅ Hugging Face 저장소 설정 (본인 username으로 변경!)
HF_REPO_ID = "YOUR_USERNAME/illegal-parking-models"

# ==================== 모델 클래스 ====================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 
                             kernel_size, padding=padding)

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
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))

class ConvLSTM_NoBN(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM_NoBN, self).__init__()
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

class ConvLSTM_BN(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM_BN, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_channels if i == 0 else hidden_channels
            self.cell_list.append(ConvLSTMCell(cur_input_dim, hidden_channels, kernel_size))
            self.bn_list.append(nn.BatchNorm2d(hidden_channels))
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
            b, s, ch, h_dim, w_dim = layer_output.size()
            layer_output = self.bn_list[layer_idx](layer_output.view(b*s, ch, h_dim, w_dim)).view(b, s, ch, h_dim, w_dim)
            cur_layer_input = layer_output
        return self.output_conv(cur_layer_input[:, -1]), None

    def _init_hidden(self, batch_size, image_size):
        return [cell.init_hidden(batch_size, image_size) for cell in self.cell_list]

class LSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, output_height, output_width, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.output_height = output_height
        self.output_width = output_width
        
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.lstm_input_dim = 32 * output_height * output_width
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_height * output_width)
        )
        
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        spatial_features = []
        for t in range(seq_len):
            feat = self.spatial_encoder(x[:, t, :, :, :])
            feat_flat = feat.reshape(batch_size, -1)
            spatial_features.append(feat_flat)
        lstm_input = torch.stack(spatial_features, dim=1)
        lstm_out, _ = self.lstm(lstm_input)
        last_hidden = lstm_out[:, -1, :]
        decoded = self.decoder(last_hidden)
        spatial_output = decoded.reshape(batch_size, 1, self.output_height, self.output_width)
        final_output = self.spatial_decoder(spatial_output)
        return final_output.squeeze(1)

# ==================== Request/Response 모델 ====================

class PredictionRequest(BaseModel):
    region: str
    model: str
    date: Optional[str] = None

# ==================== 좌표 정보 ====================

REGION_INFO = {
    'seoul': {
        'lat_range': (37.42, 37.70),
        'lon_range': (126.76, 127.18),
        'center': [37.5665, 126.978],
        'name': '서울특별시'
    },
    'haeundae': {
        'lat_range': (35.14, 35.20),
        'lon_range': (129.10, 129.22),
        'center': [35.1650, 129.1650],
        'name': '부산 해운대구'
    }
}

# ==================== Helper 함수 ====================

def grid_to_geojson(grid, lat_bins, lon_bins, threshold=0.01):
    """그리드를 GeoJSON으로 변환"""
    features = []
    H, W = grid.shape
    
    for i in range(H):
        for j in range(W):
            value = float(grid[i, j])
            
            if value < threshold:
                continue
            
            lat_min, lat_max = lat_bins[i], lat_bins[i+1]
            lon_min, lon_max = lon_bins[j], lon_bins[j+1]
            
            if value < 0.3:
                intensity = 'low'
                color = '#FEF0D9'
            elif value < 0.7:
                intensity = 'medium'
                color = '#FD8D3C'
            else:
                intensity = 'high'
                color = '#E31A1C'
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_min, lat_min],
                        [lon_max, lat_min],
                        [lon_max, lat_max],
                        [lon_min, lat_max],
                        [lon_min, lat_min]
                    ]]
                },
                "properties": {
                    "grid_id": f"grid_{i}_{j}",
                    "prediction": round(value, 4),
                    "intensity": intensity,
                    "color": color
                }
            }
            features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

def load_models():
    """Hugging Face에서 모델 다운로드 및 로드 (해운대 LSTM 제외)"""
    global models
    
    # ✅ 해운대 LSTM 제외 - 5개 모델만
    model_configs = {
        'seoul_convlstm_nobn': {
            'filename': 'seoul_convlstm_nobn.pth',
            'class': ConvLSTM_NoBN,
            'params': {'input_channels': 3, 'hidden_channels': 128, 'kernel_size': 3, 'num_layers': 3},
            'grid_size': (20, 50)
        },
        'seoul_convlstm_bn': {
            'filename': 'seoul_convlstm_bn.pth',
            'class': ConvLSTM_BN,
            'params': {'input_channels': 3, 'hidden_channels': 128, 'kernel_size': 3, 'num_layers': 3},
            'grid_size': (20, 50)
        },
        'seoul_lstm': {
            'filename': 'seoul_lstm.pth',
            'class': LSTMModel,
            'params': {'input_channels': 3, 'hidden_size': 256, 'num_layers': 3, 'output_height': 20, 'output_width': 50},
            'grid_size': (20, 50)
        },
        'haeundae_convlstm_nobn': {
            'filename': 'haeundae_convlstm_nobn.pth',
            'class': ConvLSTM_NoBN,
            'params': {'input_channels': 3, 'hidden_channels': 128, 'kernel_size': 3, 'num_layers': 3},
            'grid_size': (64, 64)
        },
        'haeundae_convlstm_bn': {
            'filename': 'haeundae_convlstm_bn.pth',
            'class': ConvLSTM_BN,
            'params': {'input_channels': 3, 'hidden_channels': 128, 'kernel_size': 3, 'num_layers': 3},
            'grid_size': (64, 64)
        },
        # ❌ 해운대 LSTM 제외 (용량 문제)
        # 'haeundae_lstm': {...}
    }
    
    os.makedirs('./models', exist_ok=True)
    
    for model_name, config in model_configs.items():
        try:
            print(f"📥 다운로드 중: {config['filename']}")
            
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=config['filename'],
                cache_dir='./models',
                resume_download=True
            )
            
            print(f"✅ 다운로드 완료: {model_path}")
            
            model = config['class'](**config['params']).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            models[model_name] = {
                'model': model,
                'grid_size': config['grid_size']
            }
            print(f"✅ {model_name} 로드 완료")
            
        except Exception as e:
            print(f"❌ {model_name} 로드 실패: {e}")
            print(f"   확인: https://huggingface.co/{HF_REPO_ID}/tree/main")

# ==================== API 엔드포인트 ====================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    print("🚀 서버 시작 중...")
    print(f"📍 Device: {device}")
    print(f"📦 Hugging Face Repo: {HF_REPO_ID}")
    print(f"⚠️  해운대 LSTM 제외 (5개 모델만 로드)")
    load_models()
    print(f"✅ {len(models)}개 모델 로드 완료")

@app.get("/")
async def root():
    """메인 페이지"""
    return FileResponse('static/index.html')

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys()),
        "device": str(device),
        "hf_repo": HF_REPO_ID
    }

@app.get("/api/regions")
async def get_regions():
    """지역 정보 반환"""
    return {
        "success": True,
        "regions": REGION_INFO
    }

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """예측 API"""
    try:
        region = request.region
        model_type = request.model
        
        if region not in REGION_INFO:
            raise HTTPException(status_code=400, detail=f"Invalid region: {region}")
        
        model_key = f"{region}_{model_type}"
        
        # ✅ 해운대 LSTM 요청 시 에러 메시지
        if model_key == "haeundae_lstm":
            raise HTTPException(
                status_code=400, 
                detail="해운대 LSTM 모델은 현재 지원하지 않습니다. ConvLSTM 모델을 사용해주세요."
            )
        
        if model_key not in models:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_key}")
        
        model_info = models[model_key]
        model = model_info['model']
        H, W = model_info['grid_size']
        
        dummy_input = torch.randn(1, 14, 3, H, W).to(device) * 0.3
        
        hotspot_count = np.random.randint(5, 15)
        for _ in range(hotspot_count):
            h_pos = np.random.randint(0, H)
            w_pos = np.random.randint(0, W)
            h_size = min(5, H - h_pos)
            w_size = min(5, W - w_pos)
            dummy_input[:, :, 0, h_pos:h_pos+h_size, w_pos:w_pos+w_size] += torch.randn(1, 14, h_size, w_size).to(device) * 0.5
        
        dummy_input = torch.clamp(dummy_input, 0, 1)
        
        with torch.no_grad():
            if 'lstm' in model_type:
                prediction = model(dummy_input)
                if prediction.dim() == 2:
                    prediction = prediction.unsqueeze(0)
            else:
                prediction, _ = model(dummy_input)
        
        pred_grid = prediction.squeeze().cpu().numpy()
        
        if pred_grid.max() > 0:
            pred_grid = pred_grid / pred_grid.max()
        
        region_info = REGION_INFO[region]
        lat_min, lat_max = region_info['lat_range']
        lon_min, lon_max = region_info['lon_range']
        
        lat_bins = np.linspace(lat_min, lat_max, H + 1)
        lon_bins = np.linspace(lon_min, lon_max, W + 1)
        
        geojson = grid_to_geojson(pred_grid, lat_bins, lon_bins)
        
        metadata = {
            'region': region_info['name'],
            'model': model_type,
            'grid_size': [H, W],
            'total_grids': int(pred_grid.size),
            'prediction_count': int((pred_grid > 0.01).sum()),
            'max_value': float(pred_grid.max()),
            'mean_value': float(pred_grid.mean()),
            'date': request.date or datetime.now().strftime('%Y-%m-%d')
        }
        
        return {
            "success": True,
            "data": geojson,
            "metadata": metadata
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if os.path.exists('static'):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
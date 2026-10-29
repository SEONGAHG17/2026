import torch
import torch.nn as nn

class WeatherPredictionLSTM(nn.Module):
    def __init__(self, input_features, hidden_dim, num_weather_classes):
        super(WeatherPredictionLSTM, self).__init__()
        
        # input_features = 4 (기온, 습도, 풍속, 강수량 4가지 수치 데이터)
        self.lstm = nn.LSTM(input_size=input_features, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        
        # 우측 하단 선형 변환 (Dense Layer)
        # LSTM이 누적해 온 날씨 흐름(hidden_dim)을 받아 최종 날씨 종류 수만큼 점수로 바꿈
        self.fc = nn.Linear(hidden_dim, num_weather_classes)

    def forward(self, x):
        # x shape: [배치 크기(지역 수), 시간 길이(최근 24시간), 4가지 날씨 수치]
        # 즉, "각 지역별로 지난 24시간 동안의 기온·습도·풍속·강수량 변화"가 쏙 들어옵니다.
        
        out, (h_n, c_n) = self.lstm(x)
        
        # 24시간 중 '가장 마지막 시간(현재)' 시점에서 나온 은닉 상태를 뽑음 ($HS_t$)
        # 24시간 동안의 날씨 흐름과 누적된 변화가 이 마지막 상태에 압축되어 담깁니다.
        latest_weather_context = out[:, -1, :] 
        
        # 우측 하단 선형 변환을 거쳐 각 날씨별 점수(로짓, Z_t)를 뽑음
        weather_scores = self.fc(latest_weather_context)
        
        return weather_scores

# --- 세팅 예시 ---
# 24시간 동안의 기상 수치 입력, LSTM 은닉 차원 32, 예측할 날씨 종류 3개(맑음, 비, 눈)
model = WeatherPredictionLSTM(input_features=4, hidden_dim=32, num_weather_classes=3)

# 가상의 기상 데이터 입력 (배치 크기 10개 지역, 24시간 기록, 4개 기상 수치)
regional_weather_batch = torch.randn(10, 24, 4) 

output = model(regional_weather_batch)
print("날씨 예측 AI의 최종 출력 형태:", output.shape) # 기대값: torch.Size([10, 3]) -> 각 지역별 [맑을 확률, 비올 확률, 눈올 확률]
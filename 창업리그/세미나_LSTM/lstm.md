# LSTM (Long Short-Term Memory) 핵심 원리 및 구조 가이드

## 1. LSTM이란?
LSTM(Long Short-Term Memory, 장단기 기억 신경망)은 순차 데이터를 처리하는 RNN 계열의 신경망입니다. **RNN이 길어지는 시퀀스 데이터에서 과거의 정보를 잊어버리는 문제(장기 의존성 문제)**를 해결하기 위해 고안되었습니다.

---

## 2. RNN의 한계: 장기 의존성 문제 (Long-Term Dependency Problem)
RNN의 Hidden State 연산은 매 Time Step마다 새로운 Weight와 활성화 함수($\tanh$)를 거쳐 계속 변환됩니다.

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

이로 인해 Sequence가 길어지면 수십 단계 전의 정보가 연산을 거치며 희석되거나 사라지며(기울기 소실, Vanishing Gradient), 오래된 과거 정보를 보존하기 어려워집니다.

---

## 3. LSTM의 핵심 아이디어: Cell State ($C_t$)
LSTM은 기존 RNN의 **Hidden State ($h_t$)** 외에 장기 기억을 담당하는 **Cell State ($C_t$)**를 별도의 통로로 추가했습니다.

| 상태 (State) | 기호 | 역할 |
|---|---|---|
| **Cell State** | $C_t$ | 컨베이어 벨트처럼 정보를 큰 변형 없이 장기적으로 전달하는 상태 |
| **Hidden State** | $h_t$ | 현재 시점의 출력 계산 및 다음 Time Step에 전달되는 단기 기억 상태 |

---

## 4. 정보 흐름 제어: 3가지 Gate
LSTM은 정보를 무조건 저장하지 않고, 3가지 게이트(Gate)를 통해 저장·삭제·출력을 선택적으로 제어합니다.

1. **Forget Gate ($f_t$)**: 이전 기억 중 무엇을 **버릴지** 결정
2. **Input Gate ($i_t$)**: 새로 들어온 정보 중 무엇을 **저장할지** 결정
3. **Output Gate ($o_t$)**: 갱신된 기억 중 무엇을 **밖으로 출력할지** 결정

### 왜 Gate에 Sigmoid ($\sigma$)를 사용하는가?
Sigmoid 함수는 출력값이 **0과 1 사이**입니다.
- `0`에 가까울수록: 정보 완전 차단 (삭제)
- `1`에 가까울수록: 정보 완전 통과 (유지)

> **Sigmoid vs $\tanh$의 역할 구분**
> - **Sigmoid ($\sigma$)**: 정보의 통과량(비율)을 조절 (0~1)
> - **$\tanh$**: 새로운 정보의 내용이나 상태 값을 생성/표현 (-1~1)

---

## 5. 한 Time Step에서의 LSTM 계산 절차

### Step 1: Forget Gate (망각 게이트)
이전 Hidden State($h_{t-1}$)와 현재 입력($x_t$)을 보고 past memory 중 버릴 비율을 계산합니다.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

### Step 2: Input Gate & Candidate Memory (입력 게이트 및 후보 기억)
새로운 정보 중 얼만큼 보관할지($i_t$) 결정하고, 새 기억의 후보가 될 내용($\tilde{C}_t$)을 생성합니다.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

### Step 3: Cell State Update (장기 기억 갱신)
과거 기억에서 버릴 것은 버리고, 새로운 기억 중 중요한 것을 더해 새로운 Cell State ($C_t$)를 만듭니다.

$$
\boxed{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}
$$

*(단, $\odot$는 원소별 곱셈인 Hadamard product를 의미합니다.)*

### Step 4: Output Gate & Hidden State (출력 게이트 및 은닉 상태)
갱신된 Cell State를 $\tanh$에 통과시켜 값의 범위를 조정하고, Output Gate의 제어에 따라 현재 시점의 Hidden State ($h_t$)를 결정합니다.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
\boxed{h_t = o_t \odot \tanh(C_t)}
$$

---

## 6. LSTM 핵심 수식 요약

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\[6pt]
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\[6pt]
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\[6pt]
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\[6pt]
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\[6pt]
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

---

## 7. LSTM 구조 종합 이해 점검 (Self-Check)

1. **왜 RNN의 Hidden State만으로는 부족한가?**
   - 연산이 반복되면서 먼 과거의 Gradient가 소실되어 중요한 과거 정보를 장기간 유지하기 힘들기 때문입니다.
2. **Cell State와 Hidden State의 역할 차이는?**
   - Cell State는 변형을 최소화하여 오랫동안 전달되는 **장기 기억 통로**이며, Hidden State는 현재 타임스텝의 **작업 및 출력용 상태**입니다.
3. **Gate의 가중치($W$)는 어떻게 학습되는가?**
   - 사람이 조건문(`if`)을 지정하는 것이 아니라, 역전파(Backpropagation) 과정에서 오차를 줄이는 방향으로 게이트의 Weight와 Bias가 스스로 업데이트됩니다.
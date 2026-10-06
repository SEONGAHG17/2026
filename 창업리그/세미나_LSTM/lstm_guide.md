# LSTM 입문 전 필수 개념 가이드: Sequence Data에서 LSTM까지

LSTM(Long Short-Term Memory)을 효과적으로 이해하기 위해서는 기초 개념부터 단계별로 차근차근 학습을 진행하는 것이 좋습니다.

```
Sequence Data ──> ANN ──> Activation Function ──> RNN ──> LSTM
```

---

## 1. 시퀀스 데이터 (Sequence Data)

### 1.1 시퀀스 데이터란?
**시퀀스 데이터(Sequence Data)**는 데이터가 일정한 순서로 나열되어 있고, 그 **순서가 데이터의 의미에 직접적인 영향**을 주는 데이터입니다. 단순한 데이터의 집합과 달리 순서가 변경되면 데이터의 의미가 손상되거나 변질됩니다.

```
10 ──> 20 ──> 30 ──> 40 ──> 50
```
- 중요한 점: 숫자 개별 값뿐만 아니라 `10 다음 20`, `20 다음 30`이라는 **순서와 흐름**이 중요합니다.

### 1.2 대표적인 시퀀스 데이터 예시

* **텍스트 (Text)**:
  - `나는` $\rightarrow$ `학교에` $\rightarrow$ `간다` (단어 순서 변경 시 문장의 의미가 달라짐)
* **음성 (Audio)**:
  - 시간에 따라 소리가 지속적으로 변화하는 일차원 연속 데이터
* **센서 데이터 (Time-series Sensor Data)**:
  - `10:00 (21℃)` $\rightarrow$ `10:01 (22℃)` $\rightarrow$ `10:02 (23℃)`
* **주가 데이터 (Stock Prices)**:
  - `월(100)` $\rightarrow$ `화(103)` $\rightarrow$ `수(101)` $\rightarrow$ `목(108)` $\rightarrow$ `금(110)`

### 1.3 Time Step (타임 스텝)
시퀀스 데이터를 처리할 때의 **각 순서상의 위치 또는 시간 단위**를 **Time Step($t$)**이라고 부릅니다.

$$\begin{aligned}
t=1 &: \text{"나는"} \\
t=2 &: \text{"학교에"} \\
t=3 &: \text{"간다"}
\end{aligned}$$

### 1.4 시퀀스 데이터의 핵심 명제
> **"현재 데이터의 의미나 값은 이전 데이터와 밀접하게 연관되어 있다."**

---

## 2. 인공 신경망 (ANN)

### 2.1 ANN이란?
**ANN(Artificial Neural Network)**은 여러 개의 인공 뉴런(Neuron)을 연결하여 데이터의 패턴을 학습하는 가장 기본적인 신경망 구조입니다.

```
[Input Layer] ──> [Hidden Layer] ──> [Output Layer]
```

### 2.2 인공 뉴런의 연산 구조
하나의 뉴런은 입력값($x$), 가중치($w$), 편향($b$)을 이용하여 계산을 수행합니다.

1. **입력 및 가중치 곱합 연산**:
   $$z = x_1 w_1 + x_2 w_2 + x_3 w_3 + b = \sum_{i} x_i w_i + b$$
2. **활성화 함수 적용**:
   $$y = f(z)$$

### 2.3 Weight(가중치)와 Bias(편향)
* **Weight ($w$)**: 각 입력 변수가 결과에 미치는 **중요도(영향력)**를 나타냅니다. 학습 과정에서 자동으로 최적화됩니다.
* **Bias ($b$)**: 뉴런의 **활성화 임계값**을 조절하여 모델의 표현력을 높여줍니다.

$$\boxed{z = wx + b}$$

### 2.4 ANN의 학습과 한계
ANN은 예측값과 실제 정답 사이의 **오차(Loss)**를 줄이는 방향으로 Weight와 Bias를 수정(역전파, Backpropagation)하며 학습합니다.

* **ANN의 한계**: 입력 데이터를 독립적인 단일 데이터로 처리하므로, **이전 단계의 정보를 기억해야 하는 시퀀스 데이터 처리에는 부적합**합니다.

---

## 3. 활성화 함수 (Activation Function)

### 3.1 활성화 함수의 역할
활성화 함수는 신경망 계산 결과에 **비선형성(Non-linearity)**을 추가합니다. 활성화 함수가 없다면 선형 연산의 중첩에 불과하여 복잡한 데이터 구조를 학습할 수 없습니다.

$$\text{입력 } (x) \longrightarrow \text{선형 연산 } (z = wx+b) \longrightarrow \text{비선형 변환 } (y = f(z))$$

### 3.2 주요 활성화 함수

#### 1) ReLU (Rectified Linear Unit)
$$ReLU(x) = \max(0, x)$$
- $x < 0$ 일 때는 $0$, $x \ge 0$ 일 때는 $x$ 그대로 출력.

#### 2) Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- 출력값을 $0$과 $1$ 사이로 압축하여, **확률값 표현**이나 Gate 제어 시 사용.

#### 3) Tanh (Hyperbolic Tangent)
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- 출력값을 $-1$과 $1$ 사이로 변환하며, RNN의 Hidden State 업데이트 시 주로 사용.

---

## 4. 순환 신경망 (RNN)

### 4.1 RNN이란?
**RNN(Recurrent Neural Network)**은 이전 Time Step의 정보(Hidden State)를 다음 Time Step으로 순환시켜 전달하는 신경망입니다.

> **핵심 개념**: 현재 입력($x_t$) + 이전 상태($h_{t-1}$) $\rightarrow$ 현재 상태($h_t$)

### 4.2 Hidden State (은닉 상태)
이전까지 입력된 데이터들의 **기억/맥락 정보**를 담고 있는 Vector입니다.

$$\begin{aligned}
\text{Step 1:} \quad &\text{"나는"} &\longrightarrow \quad &h_1 \\
\text{Step 2:} \quad &\text{"학교에"} + h_1 &\longrightarrow \quad &h_2 \quad (\text{"나는 학교에"의 정보 포함}) \\
\text{Step 3:} \quad &\text{"간다"} + h_2 &\longrightarrow \quad &h_3 \quad (\text{전체 문장 맥락 포함})
\end{aligned}$$

### 4.3 RNN 수식 및 구조

$$\boxed{h_t = \tanh(W_x x_t + W_h h_{t-1} + b)}$$

| 기호 | 의미 |
| :--- | :--- |
| $x_t$ | 현재 타임스텝의 입력 |
| $h_{t-1}$ | 이전 타임스텝의 Hidden State |
| $W_x$ | 입력 데이터용 가중치 |
| $W_h$ | 이전 Hidden State용 가중치 |
| $b$ | 편향 (Bias) |
| $h_t$ | 새로 업데이트된 현재 Hidden State |

### 4.4 펼친 RNN 구조 (Unfolded RNN)

```
       h₀             h₁             h₂             h₃
       │              │              │              │
       ▼              ▼              ▼              ▼
x₁ ──> [ RNN ] ──> x₂ ──> [ RNN ] ──> x₃ ──> [ RNN ] ──> x₄ ──> [ RNN ]
       │              │              │              │
       ▼              ▼              ▼              ▼
       h₁             h₂             h₃             h₄
```

---

## 5. RNN의 한계와 LSTM으로의 전환

### 5.1 RNN의 치명적 단점: 장기 의존성 문제 (Long-Term Dependency Problem)
RNN은 Time Step이 길어질수록 먼 과거의 정보가 기울기 손실(Vanishing Gradient) 현상으로 인해 점점 사라지게 됩니다.

> **예시**:
> *"나는 어릴 때 **한국**에서 태어났고 ... (중략) ... 지금은 **[ ? ]**어를 사용한다."*

가장 앞쪽의 `한국`이라는 정보가 먼 뒷부분까지 유지되지 못해 정확한 예측이 어려워집니다.

### 5.2 해결책: LSTM (Long Short-Term Memory)
RNN의 장기 의존성 문제를 해결하기 위해, **정보를 선택적으로 기억하거나 잊어버리는 매커니즘(Gate)**을 도입한 것이 **LSTM**입니다.

```
[Sequence Data]  ──>  [ANN]  ──>  [Activation]  ──>  [RNN]  ──>  [LSTM]
 (순서의 중요성)      (기본 연산)    (비선형성 부여)    (순환 구조)    (선택적 기억/삭제)
```

LSTM 학습 단계에서는 다음 Gate 구조들을 중점적으로 살펴보게 됩니다:
1. **Forget Gate**: 불필요한 과거 정보 삭제
2. **Input Gate**: 현재 들어온 중요한 정보 저장
3. **Cell State Update**: 장기 기억 정보 업데이트
4. **Output Gate**: 최종 출력 및 다음 Hidden State 결정
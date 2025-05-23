
# 📡 AI 기반 자율주행용 고정밀 GPS 오차 보정 기술 제안서

## 📌 제안 개요

### ✔️ 배경
도심 환경에서는 건물에 의한 신호 반사(Multipath) 및 음영(NLOS)으로 인해 GPS 신호가 왜곡되며, 수 미터 이상의 위치 오차가 발생합니다. 이는 자율주행 차량의 정밀 주행에 치명적인 영향을 미칩니다.

### ✔️ 제안 기술 요약
본 제안서는 **도로 그래프 기반의 Graph Neural Network (GNN)**와 **Transformer 기반 시계열 보정 모델**을 결합하여 GPS 오차를 실시간으로 정밀하게 보정하는 기술을 제안합니다.

---

## 🧠 핵심 기술

### 1. Graph Neural Network (GNN) 기반 위치 문맥 인식

- **도로망을 그래프 구조로 모델링**  
  - 노드: 도로 세그먼트
  - 엣지: 연결 상태 (교차로, 방향 등)
- **노드 피처 예시**  
  - 평균 속도
  - 도로 폭, 차선 수
  - 건물 밀집도
  - SNR, CN0, Multipath 비율

- **적용 모델**: `GraphSAGE`  
  - 주변 노드의 정보를 샘플링하여 노드 임베딩 생성
  - 위치 문맥을 반영한 고정밀 특징 벡터 생성

### 2. GNN + Transformer 하이브리드 보정 구조

- **GNN**: 도로 구조적 특성 이해  
- **Transformer**: 시간에 따른 위치 변화 학습  
- **MLP Head**: 최종 GPS 보정량 (Δx, Δy) 출력


## 🆚 기술 비교

| 구분              | 기존 방식 (KF 등)      | 제안 방식 (AI 기반)           |
|-------------------|------------------------|-------------------------------|
| 오차 보정 정확도  | 5~10m                  | **0.5~2.0m**                  |
| 구조적 정보 활용 | 불가능                 | **도로 그래프 기반 GNN 사용** |
| 시계열 정보 학습 | 없음                   | Transformer로 시간 흐름 학습 |
| 학습 기반         | 비학습적 (Rule 기반)   | **딥러닝 기반 학습형**         |
| 실시간 처리       | 제한적                 | Transformer 경량화로 가능     |
| 위치문맥 반영     | 불가능                 | 주변 도로/환경 정보 통합 가능 |

---

## 🗺️ 활용 분야

- 🛻 **자율주행 차량**의 고정밀 위치 추정
- 🤖 **배달 로봇 / AMR(Autonomous Mobile Robot)** 실내외 위치 보정
- 🌇 **도심 NLOS 환경**에서의 GPS 신뢰도 향상
- 🚦 **스마트시티 교통 관제 시스템**에 위치 보정 기술 적용

---

## 🧪 사용 데이터셋

- **OpenStreetMap 도로망 그래프 데이터**
- **Google Smartphone Decimeter Challenge**
- **GNSS raw log** (SNR, CN0, pseudorange 등 포함)
- **차량용 Ground Truth**: RTK 또는 LiDAR 기반 참값

---

## 📆 개발 일정 (6개월 기준)

| 단계     | 기간    | 주요 내용                                   |
|----------|---------|--------------------------------------------|
| 1단계    | 1개월   | 도로 그래프 구축, GPS 로그 수집             |
| 2단계    | 2개월   | GNN 및 Transformer 모델 설계 및 학습       |
| 3단계    | 2개월   | 실시간 보정 엔진 구현, 차량 탑재 실험      |
| 4단계    | 1개월   | 성능 평가 및 기술 보고서 작성              |

---

## ✅ 기대 효과

- 🎯 **자율주행 정밀도 향상** → 차선 수준 주행 가능
- 🛰️ **도심 GPS 불량 환경 대응** → NLOS 보정 강화
- 🔌 **외부 인프라 의존도 감소** → 차량 단독 보정 가능
- 📉 **위치 오차 감소** → 평균 오차 0.5~2.0m 달성
- 📈 **산업 확장성 확보** → 물류/배달/공공 인프라로 확장

---

## 📎 모델 상세 정보

### ▶ GraphSAGE (Inductive GNN)
- 이웃 노드 샘플링 기반 노드 임베딩 생성
- 도로망 그래프를 통해 위치 문맥 파악
- 실시간 임베딩 생성에 유리

### ▶ Transformer Encoder
- 시계열 위치 변화 학습 (이전 GPS → 현재 오차 예측)
- GNN에서 추출한 도로 문맥과 결합하여 보정량 예측

### ▶ 최종 AI 보정 구조 요약

```
    [Raw GPS]       [도로 그래프]
        │               │
        ▼               ▼
  [GPS 임베딩]     [GNN 도로 임베딩]
        │               │
        └────▶[Concatenate]◀────┘
                     │
                     ▼
            [Transformer Encoder]
                     │
                     ▼
                [MLP Head]
                     │
                     ▼
           [GPS 보정량: Δx, Δy]
```

1. GNSS Raw Data Receiver
   - 멀티밴드 GNSS (L1/L2/L5) + Multi-Constellation 수신
   - Raw pseudorange, carrier phase, SNR, Doppler 수집

2. GNSS Filter Engine
   - 위성 필터링 (SNR/AGC/위치 기반 품질 판단)
   - NLOS 감지 및 제거 (3D 지도 기반 Shadow Matching)
   - RTK/PPP 처리 엔진

3. Sensor Fusion Module (Tightly-Coupled)
   - IMU + VIO + Odometry + GNSS 통합
   - 딥러닝 기반 Drift 보정 모델 포함
   - Factor Graph 기반 SLAM 구조 (GTSAM 등)

4. Localization Core
   - 위치 추정 결과 생성 (최종 절대 위치 + 신뢰도)
   - GNSS 끊겨도 위치 유지

5. Cloud Update / V2V 공유
   - 주변 차량과 품질 정보 공유 (Federated Learning 기반)
   - 서버 또는 MEC 단 클라우드 보정 정보 수신



![image](https://github.com/user-attachments/assets/7147d54f-88f9-4114-b42d-ce75157606f5)


# CYP3A4 효소 저해율 예측 모델

**Boost up AI 2025 신약 개발 경진대회** 제출용 모델

## 🏆 모델 개요

**최종 모델**: LightGBM + GNN Stacking Ensemble

### 모델 구성
1. **LightGBM**: 분자 특징 기반 예측 (최적화된 하이퍼파라미터 적용)
2. **Graph Neural Network**: 분자 그래프 구조 기반 예측
3. **Stacking Meta Model**: Ridge 회귀로 두 모델 결합

## 📁 파일 구조

```
├── train.py                  # 학습 스크립트
├── inference.py              # 추론 스크립트
├── README.md                 # 사용 가이드
├── requirements.txt          # 라이브러리 의존성
├── train.csv                 # 학습 데이터
├── test.csv                  # 테스트 데이터
└── models/                   # 학습된 모델 저장 디렉토리
    ├── lgbm_fold_0.pkl ~ lgbm_fold_4.pkl
    ├── gnn_fold_0.pth ~ gnn_fold_4.pth
    ├── stacking_meta_model.pkl
    ├── lgbm_test_preds.npy
    └── gnn_test_preds.npy
```

## 🚀 실행 방법(코랩 기준)

### 0. Github 클론
- 다음 레포지토리 클론 진행
- https://github.com/developwho/Boostup-AI-2025-Drug-Dev

### 1. 환경 설정

```bash
# 필요한 라이브러리 설치
!pip install 'numpy<2' # 실행 후 반드시 런타임 재시작 필요
!pip install rdkit-pypi catboost torch_geometric -q
```

### 2. 학습 실행

```bash
# 전체 모델 학습
python train.py
```

**학습 과정:**
- 데이터 전처리 및 특징 추출
- LightGBM 5-Fold CV 학습
- GNN 5-Fold CV 학습  
- 스태킹 메타 모델 학습
- 모든 모델 가중치 자동 저장

### 3. 추론 실행

```bash
# 테스트 데이터 예측
python inference.py
```

# CYP3A4 효소 저해율 예측 모델

**Boost up AI 2025 신약 개발 경진대회** 제출용 모델

## 🏆 모델 개요

**최종 모델**: LightGBM + GNN Stacking Ensemble
- **Private Score 복원 가능**
- **완전한 시드 고정으로 재현성 보장**
- **5-Fold Cross Validation 적용**

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

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 필요한 라이브러리 설치
pip install -r requirements.txt
```

### 2. 학습 실행

```bash
# 전체 모델 학습 (약 2-3시간 소요)
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

**생성되는 파일:**
- `stacking_submission.csv` (최종 제출 파일)
- `lgbm_submission.csv` (LightGBM 단독 예측)
- `gnn_submission.csv` (GNN 단독 예측)

## ⚙️ 하이퍼파라미터

### LightGBM 최적 파라미터
```python
{
    'learning_rate': 0.088,
    'num_leaves': 273,
    'max_depth': 3,
    'subsample': 0.999,
    'colsample_bytree': 0.998,
    'reg_alpha': 1.56e-08,
    'reg_lambda': 4.33e-05
}
```

### GNN 최적 파라미터
```python
{
    'hidden_dim': 64,
    'num_layers': 4,
    'dropout': 0.176,
    'lr': 0.00127,
    'weight_decay': 4.08e-05
}
```

## 🔒 재현성 보장

### 시드 고정
- Python random: 42
- NumPy: 42
- PyTorch: 42
- PYTHONHASHSEED: 42
- CUDA deterministic mode 활성화

### 데이터 순서 고정
- DataFrame ID 기준 정렬
- Cross-validation 분할 고정
- DataLoader shuffle 제어

## 📊 성능 지표

- **평가 메트릭**: 0.5 × (1 - min(NRMSE, 1)) + 0.5 × PCC
- **NRMSE**: Normalized Root Mean Square Error
- **PCC**: Pearson Correlation Coefficient

## 🎯 제출 전 체크리스트

### ✅ 필수 확인사항
- [ ] `train.csv`, `test.csv` 파일 존재
- [ ] `models/` 디렉토리에 모든 모델 파일 생성 확인
- [ ] `stacking_submission.csv` 파일 생성 확인
- [ ] 예측값이 0-100 범위 내에 있는지 확인

### ✅ Private Score 복원 검증
```bash
# 동일한 환경에서 재실행 시 동일한 결과 확인
python train.py
python inference.py
# → 이전 결과와 일치하는지 확인
```

## 🛠️ 시스템 요구사항

### 최소 사양
- **RAM**: 8GB 이상
- **GPU**: CUDA 지원 (권장, CPU도 가능)
- **Python**: 3.8 이상
- **실행 시간**: 학습 2-3시간, 추론 5-10분

### 주요 라이브러리 버전
```
torch >= 1.12.0
torch-geometric >= 2.1.0
lightgbm >= 3.3.0
rdkit-pypi >= 2022.3.5
scikit-learn >= 1.1.0
pandas >= 1.4.0
numpy < 2.0.0
```

## ⚠️ 주의사항

### GPU 사용 시
- CUDA 메모리 부족 시 배치 크기 조정
- PyTorch CUDA 버전 호환성 확인

### CPU 사용 시
- 학습 시간이 크게 증가할 수 있음
- GNN 학습에 상당한 시간 소요

### 메모리 부족 시
- 배치 크기 줄이기 (32 → 16)
- 시스템 메모리 8GB 이상 권장

## 🔧 문제 해결

### 자주 발생하는 오류

**1. RDKit 관련 오류**
```bash
pip uninstall rdkit rdkit-pypi
pip install rdkit-pypi
```

**2. PyTorch Geometric 설치 오류**
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
```

**3. CUDA 메모리 부족**
```python
# train.py, inference.py에서 배치 크기 조정
batch_size=16  # 기본값 32에서 16으로 변경
```

**4. 모델 파일 없음 오류**
```bash
# 학습을 다시 실행
python train.py
```

## 📝 모델 세부사항

### 분자 특징 (LightGBM 입력)
- 기본 분자 특성: MW, LogP, TPSA 등
- 원자 개수: C, N, O, S, F, Cl, Br
- 구조적 특징: 고리 수, 방향족 고리 등
- 화학 패턴: Amide, Phenyl, Trifluoromethyl 등

### 그래프 특징 (GNN 입력)
- 노드: 원자 특징 (타입, 차수, 하이브리드화 등)
- 엣지: 결합 특징 (단일/이중/삼중결합, 방향족성 등)
- 글로벌: 분자 전체 특성

### 앙상블 전략
- Out-of-Fold 예측으로 메타 특징 생성
- Ridge 회귀로 두 모델의 예측 결합
- 최종 예측값 범위 제한 및 후처리

## 📞 지원

문제 발생 시 다음을 확인하세요:
1. 모든 파일이 올바른 위치에 있는지
2. 라이브러리 버전이 요구사항과 일치하는지
3. GPU/CUDA 설정이 올바른지
4. 충분한 메모리와 저장 공간이 있는지

---

**🏆 Boost up AI 2025 신약 개발 경진대회 제출용**  
**Private Score 복원 가능한 완전 재현 모델**
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CYP3A4 모델 전체 실행 스크립트
학습 → 추론을 순차적으로 수행
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_environment():
    """실행 환경 체크"""
    print("=== 환경 체크 ===")
    
    # 필수 파일 확인
    required_files = ['train.csv', 'test.csv', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 필수 파일이 없습니다: {missing_files}")
        return False
    
    print("✅ 필수 파일 확인 완료")
    
    # Python 버전 확인
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python 3.8 이상이 필요합니다. 현재: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 라이브러리 확인
    try:
        import torch
        import torch_geometric
        import lightgbm
        import rdkit
        import sklearn
        import pandas
        import numpy
        
        print("✅ 필수 라이브러리 확인 완료")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - PyTorch Geometric: {torch_geometric.__version__}")
        print(f"  - LightGBM: {lightgbm.__version__}")
        print(f"  - RDKit: {rdkit.__version__}")
        print(f"  - scikit-learn: {sklearn.__version__}")
        print(f"  - pandas: {pandas.__version__}")
        print(f"  - numpy: {numpy.__version__}")
        
        # GPU 확인
        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA 버전: {torch.version.cuda}")
            print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️  CUDA 사용 불가능 (CPU 모드로 실행)")
            
    except ImportError as e:
        print(f"❌ 라이브러리 누락: {e}")
        print("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False
    
    return True

def run_training():
    """학습 실행"""
    print("\n=== 모델 학습 시작 ===")
    print("예상 소요 시간: 2-3시간 (GPU) / 6-8시간 (CPU)")
    
    start_time = time.time()
    
    try:
        # train.py 실행
        result = subprocess.run([sys.executable, 'train.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ 학습 완료!")
        print(result.stdout[-500:])  # 마지막 500자만 출력
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 학습 중 오류 발생:")
        print(e.stderr)
        return False
    
    elapsed_time = time.time() - start_time
    print(f"⏰ 학습 소요 시간: {elapsed_time/3600:.1f}시간")
    
    return True

def run_inference():
    """추론 실행"""
    print("\n=== 모델 추론 시작 ===")
    print("예상 소요 시간: 5-10분")
    
    start_time = time.time()
    
    try:
        # inference.py 실행
        result = subprocess.run([sys.executable, 'inference.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ 추론 완료!")
        print(result.stdout[-500:])  # 마지막 500자만 출력
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 추론 중 오류 발생:")
        print(e.stderr)
        return False
    
    elapsed_time = time.time() - start_time
    print(f"⏰ 추론 소요 시간: {elapsed_time:.1f}초")
    
    return True

def check_results():
    """결과 파일 확인"""
    print("\n=== 결과 확인 ===")
    
    # 생성되어야 할 파일들
    expected_files = [
        'stacking_submission.csv',
        'lgbm_submission.csv', 
        'gnn_submission.csv'
    ]
    
    expected_model_files = [
        'models/stacking_meta_model.pkl',
        'models/lgbm_test_preds.npy',
        'models/gnn_test_preds.npy'
    ] + [f'models/lgbm_fold_{i}.pkl' for i in range(5)] + \
        [f'models/gnn_fold_{i}.pth' for i in range(5)]
    
    all_files = expected_files + expected_model_files
    missing_files = []
    
    for file in all_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  일부 파일이 생성되지 않았습니다: {missing_files}")
        return False
    
    print("✅ 모든 결과 파일 생성 완료")
    
    # 제출 파일 통계 확인
    try:
        import pandas as pd
        submission = pd.read_csv('stacking_submission.csv')
        
        print(f"\n📊 최종 제출 파일 통계:")
        print(f"  - 샘플 수: {len(submission)}")
        print(f"  - 평균: {submission['Inhibition'].mean():.2f}")
        print(f"  - 중앙값: {submission['Inhibition'].median():.2f}")
        print(f"  - 표준편차: {submission['Inhibition'].std():.2f}")
        print(f"  - 범위: {submission['Inhibition'].min():.2f} ~ {submission['Inhibition'].max():.2f}")
        
        # 이상값 체크
        if submission['Inhibition'].min() < 0 or submission['Inhibition'].max() > 100:
            print("⚠️  예측값이 예상 범위(0-100)를 벗어났습니다.")
        else:
            print("✅ 예측값이 정상 범위 내에 있습니다.")
            
    except Exception as e:
        print(f"⚠️  제출 파일 통계 확인 중 오류: {e}")
    
    return True

def main():
    """메인 실행 함수"""
    print("🚀 CYP3A4 모델 전체 실행 시작")
    print("=" * 50)
    
    # 1. 환경 체크
    if not check_environment():
        print("\n❌ 환경 체크 실패. 문제를 해결한 후 다시 실행하세요.")
        return
    
    # 사용자 확인
    print("\n⚠️  학습에는 많은 시간이 소요됩니다.")
    response = input("계속 진행하시겠습니까? (y/N): ").lower().strip()
    
    if response not in ['y', 'yes']:
        print("실행을 취소했습니다.")
        return
    
    total_start_time = time.time()
    
    # 2. 학습 실행
    if not run_training():
        print("\n❌ 학습 실패")
        return
    
    # 3. 추론 실행
    if not run_inference():
        print("\n❌ 추론 실패")
        return
    
    # 4. 결과 확인
    if not check_results():
        print("\n⚠️  일부 결과 파일에 문제가 있을 수 있습니다.")
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "=" * 50)
    print("🎉 모든 작업 완료!")
    print(f"⏰ 총 소요 시간: {total_elapsed/3600:.1f}시간")
    print("\n📁 생성된 주요 파일:")
    print("  - stacking_submission.csv (최종 제출 파일)")
    print("  - models/ 디렉토리 (학습된 모델들)")
    print("\n🎯 다음 단계:")
    print("  1. stacking_submission.csv를 대회 사이트에 제출")
    print("  2. Private Score 복원 검증을 위해 models/ 디렉토리 보관")
    print("  3. 필요시 코드와 함께 제출")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  사용자에 의해 실행이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        print("자세한 오류 정보는 로그를 확인하세요.")

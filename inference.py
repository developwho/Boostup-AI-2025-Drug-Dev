#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 라이브러리 임포트
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# RDKit 임포트
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

warnings.filterwarnings('ignore')

# ===== 시드 고정 =====
def seed_everything(seed=42):
    """재현성을 위해 시드 고정"""
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ 모든 Random Seed를 {seed} 값으로 고정했습니다.")

# 시드 고정
SEED = 42
seed_everything(SEED)

# ===== 데이터 로딩 =====
def load_test_data():
    """테스트 데이터 로드"""
    test_df = pd.read_csv('test.csv')
    
    # ID 기준으로 정렬하여 순서 고정
    test_df = test_df.sort_values('ID').reset_index(drop=True)
    
    print(f"테스트 데이터: {test_df.shape}")
    return test_df

# ===== 분자 특징 추출 (LightGBM용) =====
def extract_molecular_features(smiles):
    """SMILES로부터 분자 특징 추출"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        features = {}

        # 기본 분자 특성
        features['MW'] = Descriptors.ExactMolWt(mol)
        features['LogP'] = Crippen.MolLogP(mol)
        features['NumHDonors'] = Lipinski.NumHDonors(mol)
        features['NumHAcceptors'] = Lipinski.NumHAcceptors(mol)
        features['NumRotatableBonds'] = Lipinski.NumRotatableBonds(mol)
        features['NumAromaticRings'] = Lipinski.NumAromaticRings(mol)
        features['TPSA'] = Descriptors.TPSA(mol)
        features['NumHeteroatoms'] = Lipinski.NumHeteroatoms(mol)

        # 원자 수
        features['NumCarbon'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        features['NumNitrogen'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        features['NumOxygen'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        features['NumSulfur'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
        features['NumFluorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
        features['NumChlorine'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl')
        features['NumBromine'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br')

        # 구조적 특징
        features['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
        features['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
        features['FractionCSP3'] = Lipinski.FractionCSP3(mol)
        features['NumSaturatedRings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        features['NumAliphaticRings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)

        # 복잡도 지표
        features['BertzCT'] = Descriptors.BertzCT(mol)
        features['MolMR'] = Crippen.MolMR(mol)

        # 극성 지표
        features['NumPolarAtoms'] = features['NumNitrogen'] + features['NumOxygen'] + features['NumSulfur']
        features['PolarRatio'] = features['NumPolarAtoms'] / features['NumHeavyAtoms'] if features['NumHeavyAtoms'] > 0 else 0

        # SMILES 길이
        features['SmilesLength'] = len(smiles)

        # 중요 패턴
        features['HasAmide'] = int('C(=O)N' in smiles or 'NC(=O)' in smiles)
        features['HasPhenyl'] = int('c1ccccc1' in smiles)
        features['HasTrifluoromethyl'] = int('C(F)(F)F' in smiles)
        features['HasMorpholine'] = int('C1COCCN1' in smiles)
        features['HasPiperidine'] = int('C1CCNCC1' in smiles)
        features['NumBranches'] = smiles.count('(')

        return features
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {str(e)}")
        return None

def create_feature_dataframe(df):
    """데이터프레임의 모든 분자에 대해 특징 추출"""
    features_list = []

    for idx, row in df.iterrows():
        features = extract_molecular_features(row['Canonical_Smiles'])
        if features:
            features['ID'] = row['ID']
            features_list.append(features)

    return pd.DataFrame(features_list)

# ===== GNN 모델 정의 =====
class MolecularGraph:
    """SMILES를 그래프 구조로 변환"""

    def __init__(self):
        self.atom_features_dim = 0

    def atom_features(self, atom):
        """원자 특징 추출"""
        features = []

        # 원자 타입 (one-hot)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Other']
        atom_type = atom.GetSymbol() if atom.GetSymbol() in atom_types else 'Other'
        features.extend([1 if t == atom_type else 0 for t in atom_types])

        # 원자 특성
        features.extend([
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7))
        ])

        if self.atom_features_dim == 0:
            self.atom_features_dim = len(features)
        return features

    def bond_features(self, bond):
        """결합 특징 추출"""
        bond_type = bond.GetBondType()
        features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        return features

    def smiles_to_graph(self, smiles):
        """SMILES를 PyTorch Geometric Data 객체로 변환"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 노드 특징
        atom_features = [self.atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_features, dtype=torch.float)

        # 엣지 인덱스와 특징
        if mol.GetNumBonds() > 0:
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
                bond_feat = self.bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)

        # 글로벌 특징
        try:
            global_features = torch.tensor([
                Descriptors.ExactMolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                len(smiles),
                int('C(=O)N' in smiles or 'NC(=O)' in smiles),
                int('c1ccccc1' in smiles),
            ], dtype=torch.float)
        except Exception:
            global_features = torch.zeros(6, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    global_features=global_features)

class CYP3A4_GNN(nn.Module):
    """CYP3A4 예측을 위한 Graph Neural Network 구축"""

    def __init__(self, num_features, hidden_dim=64, num_layers=4, dropout=0.176):
        super(CYP3A4_GNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.conv_layers.append(GATConv(num_features, hidden_dim, heads=4, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))

        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))

        self.conv_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.global_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch, global_features = data.x, data.edge_index, data.batch, data.global_features

        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        global_features = global_features.view(-1, 6)
        global_feat = self.global_mlp(global_features)

        x = torch.cat([x_mean, x_max, global_feat], dim=1)
        out = self.predictor(x)
        return out.squeeze()

def prepare_gnn_data(df, mol_graph):
    """데이터를 GNN 형식으로 변환"""
    graphs = []
    ids = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting to graphs"):
        graph = mol_graph.smiles_to_graph(row['Canonical_Smiles'])
        if graph is not None:
            graphs.append(graph)
            ids.append(row['ID'])
    return graphs, ids

# ===== 추론 함수 =====
def inference_stacking_model():
    """학습된 스태킹 앙상블 모델로 추론 수행"""
    print("=== CYP3A4 스태킹 앙상블 모델 추론 시작 ===\n")
    
    # 모델 파일 존재 확인
    model_dir = Path('models')
    if not model_dir.exists():
        raise FileNotFoundError("models 디렉토리가 존재하지 않습니다. train.py를 먼저 실행하세요.")
    
    required_files = [
        'stacking_meta_model.pkl',
        'lgbm_test_preds.npy',
        'gnn_test_preds.npy'
    ]
    
    for file in required_files:
        if not (model_dir / file).exists():
            raise FileNotFoundError(f"필수 모델 파일이 없습니다: {file}")
    
    # 테스트 데이터 로드
    test_df = load_test_data()
    
    print("1. 저장된 테스트 예측 결과 로드 중...")
    
    # 방법 1: 이미 계산된 테스트 예측 결과 사용 (빠름)
    try:
        lgbm_test_preds = np.load('models/lgbm_test_preds.npy')
        gnn_test_preds = np.load('models/gnn_test_preds.npy')
        print("✅ 저장된 테스트 예측 결과 로드 완료")
        
    except FileNotFoundError:
        print("저장된 예측 결과가 없습니다. 모델을 직접 로드하여 예측을 수행합니다...")
        
        # 방법 2: 모델을 직접 로드해서 예측 (완전한 재현성을 위해)
        lgbm_test_preds, gnn_test_preds = inference_from_models(test_df)
    
    print(f"LightGBM 테스트 예측 shape: {lgbm_test_preds.shape}")
    print(f"GNN 테스트 예측 shape: {gnn_test_preds.shape}")
    
    # 2. stacking 메타 모델 로드
    print("\n2. 스태킹 메타 모델 로드 중...")
    with open('models/stacking_meta_model.pkl', 'rb') as f:
        meta_model = pickle.load(f)
    print("✅ 메타 모델 로드 완료")
    
    # 3. 메타 특징 생성 및 최종 예측
    print("\n3. 최종 스태킹 예측 수행 중...")
    X_meta_test = np.column_stack((lgbm_test_preds, gnn_test_preds))
    final_predictions = meta_model.predict(X_meta_test)
    
    # 4. 예측 결과 통계
    print(f"\n=== 예측 결과 통계 ===")
    print(f"평균: {np.mean(final_predictions):.2f}")
    print(f"중앙값: {np.median(final_predictions):.2f}")
    print(f"표준편차: {np.std(final_predictions):.2f}")
    print(f"최소값: {np.min(final_predictions):.2f}")
    print(f"최대값: {np.max(final_predictions):.2f}")
    
    # 5. 제출 파일 생성
    print("\n4. 제출 파일 생성 중...")
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Inhibition': final_predictions
    })
    
    submission.to_csv('stacking_submission.csv', index=False)
    print("✅ stacking_submission.csv 생성 완료!")
    
    # 6. 개별 모델 예측 결과도 저장 (비교용)
    lgbm_submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Inhibition': lgbm_test_preds
    })
    lgbm_submission.to_csv('lgbm_submission.csv', index=False)
    
    gnn_submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Inhibition': gnn_test_preds
    })
    gnn_submission.to_csv('gnn_submission.csv', index=False)
    
    print("\n=== 추론 완료 ===")
    print("생성된 파일:")
    print("  - stacking_submission.csv (최종 제출 파일)")
    print("  - lgbm_submission.csv (LightGBM 단독)")
    print("  - gnn_submission.csv (GNN 단독)")
    
    return final_predictions

def inference_from_models(test_df):
    """모델을 직접 로드해서 테스트 예측 수행 (완전한 재현성을 위해)"""
    print("모델 파일에서 직접 예측을 수행합니다...")
    
    # === LightGBM 예측 ===
    print("LightGBM 모델들 로드 및 예측 중...")
    
    # 특징 추출
    test_features = create_feature_dataframe(test_df)
    feature_cols = [col for col in test_features.columns if col != 'ID']
    X_test = test_features[feature_cols]
    
    lgbm_preds = []
    for fold in range(5):
        model_path = f'models/lgbm_fold_{fold}.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            lgbm_preds.append(model.predict(X_test))
        else:
            raise FileNotFoundError(f"LightGBM 모델 파일이 없습니다: {model_path}")
    
    lgbm_test_preds = np.mean(lgbm_preds, axis=0)
    
    # === GNN 예측 ===
    print("GNN 모델들 로드 및 예측 중...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 그래프 데이터 준비
    mol_graph = MolecularGraph()
    test_graphs, test_ids = prepare_gnn_data(test_df, mol_graph)
    num_features = mol_graph.atom_features_dim
    
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    gnn_preds = []
    for fold in range(5):
        model_path = f'models/gnn_fold_{fold}.pth'
        if os.path.exists(model_path):
            # 모델 초기화 및 로드
            model = CYP3A4_GNN(
                num_features=num_features,
                hidden_dim=64,
                num_layers=4,
                dropout=0.17649839774921566,
                lr=0.001269320542711689,
                weight_decay=4.088394611268561e-05
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # 예측 수행
            fold_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    fold_preds.extend(model(batch).cpu().numpy())
            
            gnn_preds.append(fold_preds)
        else:
            raise FileNotFoundError(f"GNN 모델 파일이 없습니다: {model_path}")
    
    gnn_test_preds = np.mean(gnn_preds, axis=0)
    
    return lgbm_test_preds, gnn_test_preds

if __name__ == "__main__":
    try:
        final_predictions = inference_stacking_model()
        print(f"\n 추론 성공 최종 예측 완료")
        print(f"예측 샘플 수: {len(final_predictions)}")
        
    except Exception as e:
        print(f"추론 중 오류 발생: {str(e)}")
        print("train.py를 먼저 실행했는지 확인하세요.")
        raise

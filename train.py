#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CYP3A4 효소 저해율 예측 모델 - 학습 스크립트
최종 모델: LightGBM + GNN Stacking Ensemble
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 라이브러리 불러오기
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

warnings.filterwarnings('ignore')

# ===== 시드 고정 =====
def seed_everything(seed=42):
    """재현을 위해 시드 고정"""
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
    print(f"모든 Random Seed를 {seed} 값으로 고정했습니다.")

# 시드 고정
SEED = 42
seed_everything(SEED)

# ===== 최적 하이퍼파라미터 =====
LGBM_PARAMS = {
    'learning_rate': 0.08805472577903849,
    'num_leaves': 273,
    'max_depth': 3,
    'subsample': 0.9987957178315473,
    'colsample_bytree': 0.9977860360368774,
    'reg_alpha': 1.5630898694424702e-08,
    'reg_lambda': 4.333580498007759e-05
}

GNN_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 4,
    'dropout': 0.17649839774921566,
    'lr': 0.001269320542711689,
    'weight_decay': 4.088394611268561e-05
}

# ===== 데이터 로딩 =====
def load_data():
    """데이터 로드 및 기본 전처리"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # ID 기준으로 정렬하여 순서 고정
    train_df = train_df.sort_values('ID').reset_index(drop=True)
    test_df = test_df.sort_values('ID').reset_index(drop=True)

    print(f"학습 데이터: {train_df.shape}")
    print(f"테스트 데이터: {test_df.shape}")
    print(f"타겟 통계 - 평균: {train_df['Inhibition'].mean():.2f}, "
          f"중앙값: {train_df['Inhibition'].median():.2f}, "
          f"표준편차: {train_df['Inhibition'].std():.2f}")

    return train_df, test_df

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
            if 'Inhibition' in row:
                features['Inhibition'] = row['Inhibition']
            features_list.append(features)

    return pd.DataFrame(features_list)

# ===== 평가 메트릭 =====
def calculate_metrics(y_true, y_pred):
    """대회 평가 메트릭 계산"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = y_true.max() - y_true.min()
    nrmse = rmse / y_range if y_range > 0 else rmse
    pcc, _ = pearsonr(y_true, y_pred)
    score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * pcc
    
    return {
        'RMSE': rmse,
        'NRMSE': nrmse,
        'PCC': pcc,
        'Score': score
    }

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
    """CYP3A4 예측을 위한 Graph Neural Network"""

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
            if 'Inhibition' in row:
                graph.y = torch.tensor([row['Inhibition']], dtype=torch.float)
            graphs.append(graph)
            ids.append(row['ID'])
    return graphs, ids

# ===== GNN 학습 함수 =====
def train_gnn_model(model, train_loader, val_loader, optimizer, criterion, device, fold, epochs=150):
    """GNN 모델 학습"""
    best_val_score = -float('inf')
    patience = 20
    patience_counter = 0
    best_model_path = f'models/gnn_fold_{fold}_best_temp.pth'

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.squeeze())

            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss = loss + 1e-5 * l2_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_preds.extend(model(batch).cpu().numpy())
                val_targets.extend(batch.y.squeeze().cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        if not np.all(np.isfinite(val_preds)):
            continue

        rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        y_range = val_targets.max() - val_targets.min()
        nrmse = rmse / y_range if y_range > 0 else rmse
        try:
            pcc = np.corrcoef(val_preds, val_targets)[0, 1]
            if np.isnan(pcc): pcc = 0
        except Exception:
            pcc = 0

        score = 0.5 * (1 - min(nrmse, 1)) + 0.5 * pcc

        if score > best_val_score:
            best_val_score = score
            patience_counter = 0
            # 항상 동일한 파일명으로 best model 저장
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    return best_val_score, best_model_path

# ===== 메인 학습 함수 =====
def train_stacking_model():
    """최종 스태킹 앙상블 모델 학습"""
    print("=== CYP3A4 스태킹 앙상블 모델 학습 시작 ===\n")
    
    # 모델 저장 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 데이터 로드
    train_df, test_df = load_data()
    
    # === PART 1: LightGBM 특징 준비 ===
    print("\n1. LightGBM 특징 추출 중...")
    train_features = create_feature_dataframe(train_df)
    test_features = create_feature_dataframe(test_df)
    
    feature_cols = [col for col in train_features.columns if col not in ['ID', 'Inhibition']]
    X_train = train_features[feature_cols]
    y_train = train_features['Inhibition']
    X_test = test_features[feature_cols]
    
    print(f"추출된 특징 수: {len(feature_cols)}")
    
    # === PART 2: GNN 데이터 준비 ===
    print("\n2. GNN 그래프 데이터 준비 중...")
    mol_graph = MolecularGraph()
    train_graphs, _ = prepare_gnn_data(train_df, mol_graph)
    test_graphs, test_ids = prepare_gnn_data(test_df, mol_graph)
    num_features = mol_graph.atom_features_dim
    
    print(f"변환된 학습 그래프: {len(train_graphs)}")
    print(f"변환된 테스트 그래프: {len(test_graphs)}")
    print(f"원자 특징 차원: {num_features}")
    
    # === PART 3: 5-Fold Cross Validation 설정 ===
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # Out-of-Fold 예측을 위한 배열 초기화
    lgbm_oof = np.zeros(len(X_train))
    gnn_oof = np.zeros(len(train_graphs))
    lgbm_test_preds = []
    gnn_test_preds = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n사용 디바이스: {device}")
    print(f"LightGBM 데이터 수: {len(X_train)}")
    print(f"GNN 그래프 수: {len(train_graphs)}")
    
    # === PART 4: 5-Fold 학습 ===
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n===== Fold {fold + 1} =====")
        
        # --- LightGBM 학습 ---
        print(f"LightGBM Fold {fold+1} 학습 중...")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 최적 파라미터로 LightGBM 설정
        lgbm_params = LGBM_PARAMS.copy()
        lgbm_params.update({
            'objective': 'regression_l1',
            'metric': 'rmse',
            'random_state': SEED,
            'n_estimators': 2000,
            'verbosity': -1
        })
        
        if torch.cuda.is_available():
            lgbm_params['device'] = 'gpu'
        
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        lgbm_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # OOF 예측 및 모델 저장
        lgbm_oof[val_idx] = lgbm_model.predict(X_val)
        lgbm_test_preds.append(lgbm_model.predict(X_test))
        
        with open(f'models/lgbm_fold_{fold}.pkl', 'wb') as f:
            pickle.dump(lgbm_model, f)
        
        val_score = calculate_metrics(y_val, lgbm_oof[val_idx])
        print(f"LightGBM Fold {fold+1} Score: {val_score['Score']:.4f}")
        
        # --- GNN 학습 (동일한 fold 인덱스 사용하되, 그래프 개수만큼만) ---
        print(f"GNN Fold {fold+1} 학습 중...")
        
        # 그래프 인덱스 조정 (그래프 수가 특징 수보다 적을 수 있음)
        gnn_train_idx = train_idx[train_idx < len(train_graphs)]
        gnn_val_idx = val_idx[val_idx < len(train_graphs)]
        
        if len(gnn_train_idx) == 0 or len(gnn_val_idx) == 0:
            print(f"⚠️  GNN Fold {fold+1}: 유효한 인덱스가 없어 건너뜀")
            continue
        
        train_data = [train_graphs[i] for i in gnn_train_idx]
        val_data = [train_graphs[i] for i in gnn_val_idx]
        
        # DataLoader 생성 (shuffle=False로 순서 고정)
        train_indices = np.arange(len(train_data))
        np.random.shuffle(train_indices)  # 시드가 고정되어 동일한 순서
        train_data_shuffled = [train_data[i] for i in train_indices]
        
        train_loader = DataLoader(train_data_shuffled, batch_size=32, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # GNN 모델 초기화
        gnn_model = CYP3A4_GNN(
            num_features=num_features,
            hidden_dim=GNN_PARAMS['hidden_dim'],
            num_layers=GNN_PARAMS['num_layers'],
            dropout=GNN_PARAMS['dropout']
        ).to(device)
        
        # 가중치 초기화 (재현성을 위해)
        torch.manual_seed(SEED + fold)
        for name, param in gnn_model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        
        optimizer = torch.optim.Adam(
            gnn_model.parameters(), 
            lr=GNN_PARAMS['lr'], 
            weight_decay=GNN_PARAMS['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # GNN 학습 실행
        best_score, best_model_path = train_gnn_model(gnn_model, train_loader, val_loader, optimizer, criterion, device, fold)
        
        # 최고 성능 모델 로드 및 저장
        gnn_model.load_state_dict(torch.load(best_model_path))
        torch.save(gnn_model.state_dict(), f'models/gnn_fold_{fold}.pth')
        
        # OOF 예측
        gnn_model.eval()
        val_fold_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_fold_preds.extend(gnn_model(batch).cpu().numpy())
        gnn_oof[gnn_val_idx] = val_fold_preds
        
        # validation 타겟값 추출 (GNN용)
        val_targets = [train_graphs[i].y.item() for i in gnn_val_idx]
        
        # 테스트 예측
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        fold_test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                fold_test_preds.extend(gnn_model(batch).cpu().numpy())
        gnn_test_preds.append(fold_test_preds)
        
        val_targets_np = np.array(val_targets)
        val_fold_preds_np = np.array(val_fold_preds)
        
        val_score = calculate_metrics(val_targets_np, val_fold_preds_np) # 수정된 변수로 호출
        print(f"GNN Fold {fold+1} Score: {val_score['Score']:.4f}")
        
        # 임시 파일 정리
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
    
    # === PART 5: 스태킹 메타 모델 학습 ===
    print("\n=== 스태킹 메타 모델 학습 ===")
    
    # 두 OOF 예측의 길이를 맞춤 (짧은 쪽에 맞춤)
    min_length = min(len(lgbm_oof), len(gnn_oof))
    
    if len(lgbm_oof) != len(gnn_oof):
        print(f"OOF 길이 불일치: LightGBM {len(lgbm_oof)}, GNN {len(gnn_oof)}")
        print(f"최소 길이 {min_length}로 맞춤")
        
        lgbm_oof_trimmed = lgbm_oof[:min_length]
        gnn_oof_trimmed = gnn_oof[:min_length]
        y_train_trimmed = y_train.iloc[:min_length]
    else:
        lgbm_oof_trimmed = lgbm_oof
        gnn_oof_trimmed = gnn_oof  
        y_train_trimmed = y_train
    
    # OOF 예측으로 메타 특징 생성
    X_meta = np.column_stack((lgbm_oof_trimmed, gnn_oof_trimmed))
    
    # Ridge 회귀로 메타 모델 학습
    meta_model = Ridge(alpha=1.0, random_state=SEED)
    meta_model.fit(X_meta, y_train_trimmed)
    
    # 메타 모델 저장
    with open('models/stacking_meta_model.pkl', 'wb') as f:
        pickle.dump(meta_model, f)
    
    # === PART 6: 최종 성능 평가 ===
    stacking_oof = meta_model.predict(X_meta)
    final_score = calculate_metrics(y_train_trimmed, stacking_oof)
    
    print(f"\n=== 최종 성능 ===")
    print(f"메타 학습 데이터 길이: {len(y_train_trimmed)}")
    print(f"LightGBM OOF Score: {calculate_metrics(y_train_trimmed, lgbm_oof_trimmed)['Score']:.4f}")
    print(f"GNN OOF Score: {calculate_metrics(y_train_trimmed, gnn_oof_trimmed)['Score']:.4f}")
    print(f"Stacking OOF Score: {final_score['Score']:.4f}")
    print(f"  - NRMSE: {final_score['NRMSE']:.4f}")
    print(f"  - PCC: {final_score['PCC']:.4f}")
    print(f"  - RMSE: {final_score['RMSE']:.4f}")
    
    # === PART 7: 테스트 예측 결과 저장 ===
    final_lgbm_test = np.mean(lgbm_test_preds, axis=0)
    final_gnn_test = np.mean(gnn_test_preds, axis=0)
    
    # 테스트 예측 저장 (inference.py에서 사용)
    np.save('models/lgbm_test_preds.npy', final_lgbm_test)
    np.save('models/gnn_test_preds.npy', final_gnn_test)
    
    print("\n 모든 모델 학습 및 저장 완료")
    print("저장된 파일:")
    print("  - models/lgbm_fold_0.pkl ~ lgbm_fold_4.pkl")
    print("  - models/gnn_fold_0.pth ~ gnn_fold_4.pth")
    print("  - models/stacking_meta_model.pkl")
    print("  - models/lgbm_test_preds.npy")
    print("  - models/gnn_test_preds.npy")

if __name__ == "__main__":
    train_stacking_model()

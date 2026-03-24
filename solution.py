"""
헬스케어 데이터 기반 투표 여부(voted) 예측
태스크: 이진분류 (voted=1 → 1, voted=2 → 0)
평가지표: ROC-AUC
모델: MLP (BatchNorm + Dropout + AdamW + CosineAnnealing)
전략: 5-Fold StratifiedKFold 앙상블
"""

# ============================================================
# 0. 라이브러리
# ============================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 1. 데이터 로드
# ============================================================
train = pd.read_csv('train.csv')
test  = pd.read_csv('test_x.csv')
sub   = pd.read_csv('sample_submission.csv')

print(f"Train: {train.shape} | Test: {test.shape}")

# ============================================================
# 2. 타깃 생성
# ============================================================
# voted: 1 = 투표함(양성), 2 = 투표안함(음성) → 0/1 이진화
train['target'] = (train['voted'] == 1).astype(int)
y = train['target'].values
print(f"클래스 분포 — 투표함(1): {y.sum():,} ({y.mean():.1%}) | "
      f"투표안함(0): {(1-y).sum():,} ({1-y.mean():.1%})")

# ============================================================
# 3. 피처 엔지니어링
# ============================================================
def feature_engineering(df):
    df = df.copy()

    qa_cols = [c for c in df.columns if c.startswith('Q') and c.endswith('A')]
    qe_cols = [c for c in df.columns if c.startswith('Q') and c.endswith('E')]
    tp_cols = [f'tp0{i}' for i in range(1, 10)] + ['tp10']
    wf_cols = ['wf_01', 'wf_02', 'wf_03']
    wr_cols = [c for c in df.columns if c.startswith('wr_')]

    # --- 설문 응답(QA): 1~5 Likert ---
    # 합계/평균/표준편차/극단 응답 비율
    df['qa_sum']     = df[qa_cols].sum(axis=1)
    df['qa_mean']    = df[qa_cols].mean(axis=1)
    df['qa_std']     = df[qa_cols].std(axis=1)
    df['qa_min']     = df[qa_cols].min(axis=1)
    df['qa_max']     = df[qa_cols].max(axis=1)
    df['qa_extreme'] = ((df[qa_cols] == 1) | (df[qa_cols] == 5)).sum(axis=1) / len(qa_cols)
    # 중간값(3) 응답 비율: 회피 성향 지표
    df['qa_neutral'] = (df[qa_cols] == 3).sum(axis=1) / len(qa_cols)

    # --- 응답시간(QE): 로그 변환으로 이상치 완화 ---
    # 이상치(0 이하 또는 비정상 큰 값)는 log1p로 자연스럽게 압축
    for c in qe_cols:
        df[c] = np.log1p(df[c].clip(lower=0))
    df['qe_mean']  = df[qe_cols].mean(axis=1)
    df['qe_std']   = df[qe_cols].std(axis=1)
    df['qe_total'] = df[qe_cols].sum(axis=1)
    # 응답이 매우 빠른 비율 (로그 변환 후 작은 값 = 빠른 응답)
    df['qe_fast_ratio'] = (df[qe_cols] < df[qe_cols].median().median()).sum(axis=1) / len(qe_cols)

    # --- Dark Triad (tp01~tp10, 0~7점) ---
    df['tp_sum']    = df[tp_cols].sum(axis=1)
    df['tp_mean']   = df[tp_cols].mean(axis=1)
    df['tp_std']    = df[tp_cols].std(axis=1)
    # tp10: 가장 타깃과 상관 높은 항목 단독 추출
    df['tp10_high'] = (df['tp10'] >= 5).astype(int)

    # --- wf / wr (이진 체크박스 그룹) ---
    df['wf_sum']  = df[wf_cols].sum(axis=1)
    df['wr_sum']  = df[wr_cols].sum(axis=1)
    df['wr_mean'] = df[wr_cols].mean(axis=1)

    # --- 범주형 인코딩 ---
    # age_group → 순서형 (연령이 높을수록 투표율 높음: 강한 신호)
    age_order = {'10s': 0, '20s': 1, '30s': 2, '40s': 3,
                 '50s': 4, '60s': 5, '+70s': 6}
    df['age_ord'] = df['age_group'].map(age_order)

    # gender → 이진 (Male=1, Female=0, 기타=2)
    df['gender_bin'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(2).astype(int)

    # race / religion → Label Encoding
    for col in ['race', 'religion']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    return df


train_fe = feature_engineering(train)
test_fe  = feature_engineering(test)

# 학습에 사용할 피처 선택
DROP = ['index', 'voted', 'target', 'age_group', 'gender', 'race', 'religion']
feature_cols = [c for c in train_fe.columns if c not in DROP]
print(f"최종 피처 수: {len(feature_cols)}")

X      = train_fe[feature_cols].values.astype(np.float32)
X_test = test_fe[feature_cols].values.astype(np.float32)

# ============================================================
# 4. 스케일링
# ============================================================
scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X)
X_test_sc   = scaler.transform(X_test)

# ============================================================
# 5. MLP 모델 정의
# ============================================================
class MLP(nn.Module):
    """
    구성: BatchNorm → Dense → ReLU → Dropout 블록 × N층
    선택 이유:
    - BatchNorm: 레이어 간 Internal Covariate Shift 완화 → 학습 안정화
    - AdamW + CosineAnnealing: 과적합 억제 + 학습률 부드럽게 감소
    - Dropout: 앙상블 효과 (학습 시 랜덤 뉴런 비활성)
    """
    def __init__(self, input_dim, hidden_dims=(512, 256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            dr = dropout if i < len(hidden_dims) - 1 else dropout * 0.5
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dr)
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# 6. 학습 / 평가 함수
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        # Gradient clipping: 폭발적 기울기 방지
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
        preds.extend(prob)
        labels.extend(yb.numpy())
    return np.array(preds), np.array(labels)


# ============================================================
# 7. K-Fold 교차검증 + 앙상블
# ============================================================
N_FOLDS    = 5
EPOCHS     = 100
BATCH_SIZE = 512
LR         = 1e-3
PATIENCE   = 15   # Early stopping patience

skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds  = np.zeros(len(X_scaled))
test_preds = np.zeros(len(X_test_sc))
fold_aucs  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\n{'─'*50}")
    print(f"  Fold {fold+1}/{N_FOLDS}")

    # DataLoader 구성
    def make_loader(X_, y_, shuffle):
        ds = TensorDataset(torch.FloatTensor(X_), torch.FloatTensor(y_))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          drop_last=shuffle)

    tr_loader  = make_loader(X_scaled[tr_idx],  y[tr_idx].astype(np.float32), True)
    val_loader = make_loader(X_scaled[val_idx],  y[val_idx].astype(np.float32), False)
    te_loader  = make_loader(X_test_sc, np.zeros(len(X_test_sc), np.float32), False)

    # 모델 초기화
    model     = MLP(X_scaled.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_val_pred, no_improve = 0, None, 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, tr_loader, optimizer, criterion)
        val_pred, val_true = predict(model, val_loader)
        val_auc = roc_auc_score(val_true, val_pred)
        scheduler.step()

        if val_auc > best_auc:
            best_auc, best_val_pred = val_auc, val_pred.copy()
            torch.save(model.state_dict(), f'/tmp/fold{fold}_best.pt')
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"    epoch {epoch:3d} | loss={tr_loss:.4f} | auc={val_auc:.4f} | best={best_auc:.4f}")

        if no_improve >= PATIENCE:
            print(f"    → Early stopping at epoch {epoch} (best={best_auc:.4f})")
            break

    # 베스트 모델로 테스트 예측
    model.load_state_dict(torch.load(f'/tmp/fold{fold}_best.pt', map_location=device))
    te_pred, _ = predict(model, te_loader)

    oof_preds[val_idx] = best_val_pred
    test_preds         += te_pred / N_FOLDS
    fold_aucs.append(best_auc)
    print(f"  Fold {fold+1} Best AUC: {best_auc:.4f}")

print(f"\n{'='*50}")
oof_auc = roc_auc_score(y, oof_preds)
print(f"  OOF AUC : {oof_auc:.4f}")
print(f"  Fold AUC: {[round(a, 4) for a in fold_aucs]}")
print(f"  Mean±Std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# ============================================================
# 8. 제출 파일 생성
# ============================================================
# sample_submission의 voted 컬럼에 양성 클래스(투표함) 확률 기입
sub['voted'] = test_preds
sub.to_csv('submission.csv', index=False)
print("\n✅ 제출 파일 저장: submission.csv")
print(sub.head())
"""
train_model.py — Treina e salva o modelo Random Forest para predição de risco.

Execute este script UMA VEZ antes de rodar o app Streamlit:
    python models/train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ── Caminhos ────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, '..', 'data')

# ── Carregamento ────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_PATH, 'pede_consolidado.csv'))
print(f"Dataset carregado: {df.shape}")

# ── Feature Engineering ─────────────────────────────────────────────
FEATURES_BASE = ['Fase', 'INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'IPP']

df['FE_IEG_x_IDA']    = df['IEG'] * df['IDA']
df['FE_Delta_IAA_IDA']= df['IAA'] - df['IDA']
df['FE_IPS_baixo']    = (df['IPS'] < 5.0).astype(float)
df['FE_IEG_baixo']    = (df['IEG'] < 5.0).astype(float)
df['FE_IDA_baixo']    = (df['IDA'] < 5.0).astype(float)
df['FE_Tres_baixos']  = df['FE_IPS_baixo'] + df['FE_IEG_baixo'] + df['FE_IDA_baixo']
df['FE_IAN_defas']    = (df['IAN'] < 10.0).astype(float)

FEATURES_ENG = [
    'FE_IEG_x_IDA', 'FE_Delta_IAA_IDA', 'FE_IPS_baixo',
    'FE_IEG_baixo', 'FE_IDA_baixo', 'FE_Tres_baixos', 'FE_IAN_defas'
]
ALL_FEATURES = FEATURES_BASE + FEATURES_ENG
TARGET = 'EM_RISCO'

# Filtra registros com features base disponíveis
base_required = [f for f in FEATURES_BASE if f != 'IPP']
df_model = df[ALL_FEATURES + [TARGET]].copy()
df_model = df_model.dropna(subset=base_required + [TARGET])

print(f"Dataset para modelagem: {df_model.shape}")
print(f"Balanceamento: {df_model[TARGET].value_counts().to_dict()}")

# ── Separação treino/teste ───────────────────────────────────────────
X = df_model[ALL_FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Modelo ───────────────────────────────────────────────────────────
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTreinando Random Forest...")
pipeline.fit(X_train, y_train)

# ── Avaliação ────────────────────────────────────────────────────────
y_pred      = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"\n{'='*50}")
print(f"  AUC-ROC CV (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  AUC-ROC Teste:       {auc:.4f}")
print(f"{'='*50}")
print(classification_report(y_test, y_pred, target_names=['Sem risco', 'Em risco']))

# ── Export ───────────────────────────────────────────────────────────
os.makedirs(BASE, exist_ok=True)

model_path    = os.path.join(BASE, 'random_forest_risco.pkl')
features_path = os.path.join(BASE, 'features.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

with open(features_path, 'wb') as f:
    pickle.dump(ALL_FEATURES, f)

print(f"\n✅ Modelo salvo em:   {model_path}")
print(f"✅ Features salvas em: {features_path}")
print("\nAgora você pode rodar o app Streamlit:")
print("  cd app && streamlit run app.py")

# 🎓 Passos Mágicos — Datathon POSTECH Fase 5

Análise de dados e modelo preditivo de risco de defasagem educacional para a Associação Passos Mágicos.

---

## 📁 Estrutura do Projeto

```
passos_magicos/
├── data/
│   ├── BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv
│   ├── BASE DE DADOS PEDE 2024 - DATATHON - PEDE2023.csv
│   ├── BASE DE DADOS PEDE 2024 - DATATHON - PEDE2024.csv
│   ├── pede_consolidado.csv          ← dataset limpo unificado
│   ├── pede_ml.csv                   ← dataset para ML
│   └── Base de dados - Passos Mágicos/  ← base relacional
├── notebooks/
│   ├── 01_leitura_limpeza.ipynb      ← limpeza e consolidação
│   ├── 02_analise_exploratoria.ipynb ← EDA + 11 perguntas
│   └── 03_modelo_preditivo.ipynb     ← Random Forest + avaliação
├── models/
│   ├── train_model.py                ← script de treino
│   ├── random_forest_risco.pkl       ← modelo treinado (gerado pelo script)
│   └── features.pkl                  ← lista de features
├── app/
│   └── app.py                        ← Streamlit app
├── requirements.txt
└── README.md
```

---

## 🚀 Como Rodar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo (obrigatório antes do app)
```bash
python models/train_model.py
```

### 3. Rodar o app Streamlit localmente
```bash
cd app
streamlit run app.py
```

### 4. Rodar os notebooks
Abra os notebooks na ordem (01 → 02 → 03) no Jupyter ou VS Code.

---

## 📊 Notebooks

| Notebook | Conteúdo |
|----------|----------|
| `01_leitura_limpeza` | Leitura dos CSVs PEDE 2022/23/24, limpeza, padronização de colunas e consolidação |
| `02_analise_exploratoria` | Análise das 11 perguntas do datathon com visualizações |
| `03_modelo_preditivo` | Feature engineering, Random Forest, AUC-ROC, feature importance, export do modelo |

---

## 🤖 Modelo Preditivo

**Algoritmo:** Random Forest (300 árvores, `class_weight='balanced'`)

**Variável-alvo:** `EM_RISCO = 1` quando Pedra == QUARTZO ou Defasagem > 0

**Features (16):**
- Base: Fase, INDE, IAA, IEG, IPS, IDA, IPV, IAN, IPP
- Engineered: IEG×IDA, IAA−IDA, flags de baixo desempenho/engajamento/psicossocial, flag defasagem

---

## 🌐 Deploy Streamlit Community Cloud

1. Faça fork/push deste repositório para o seu GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app**
4. Selecione o repositório, branch `main` e o arquivo `app/app.py`
5. Clique em **Deploy**

> ⚠️ **Importante:** o arquivo `models/random_forest_risco.pkl` precisa estar no repositório (commitar após rodar `train_model.py`).

---

## 📋 Perguntas Respondidas

| # | Indicador | Status |
|---|-----------|--------|
| 1 | IAN — perfil de defasagem | ✅ |
| 2 | IDA — tendência de desempenho | ✅ |
| 3 | IEG × IDA × IPV | ✅ |
| 4 | IAA vs desempenho real | ✅ |
| 5 | IPS e quedas futuras | ✅ |
| 6 | IPP × IAN | ✅ |
| 7 | Drivers do IPV | ✅ |
| 8 | Multidimensionalidade → INDE | ✅ |
| 9 | Modelo preditivo de risco | ✅ |
| 10 | Efetividade do programa | ✅ |
| 11 | Insights criativos | ✅ |

---

## 👥 Equipe

Desenvolvido para o **Datathon POSTECH — Fase 5**

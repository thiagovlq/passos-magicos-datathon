# 🎓 Passos Mágicos — Datathon POSTECH Fase 5

## 🌐 App Online
👉 **[Acesse o Preditor de Risco aqui](https://paapps-magicos-datathon.streamlit.app/)**

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
│   └── pede_ml.csv                   ← dataset para ML
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

**Performance:**
- AUC-ROC: 0.9733 (teste) | 0.9762 ± 0.0078 (CV 5-fold)
- Recall: 87% dos alunos em risco identificados
- Acurácia: 95%
- Threshold recomendado: 0.4 (captura 90% dos casos em risco)

**Features (16):**
- Base: Fase, INDE, IAA, IEG, IPS, IDA, IPV, IAN, IPP
- Engineered: IEG×IDA, IAA−IDA, flags de baixo desempenho/engajamento/psicossocial, flag defasagem

---

## 🌐 Deploy Streamlit Community Cloud

1. Faça fork/push deste repositório para o seu GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app**
4. Selecione o repositório, branch `master` e o arquivo `app/app.py`
5. Clique em **Deploy**

> ⚠️ **Importante:** o arquivo `models/random_forest_risco.pkl` precisa estar no repositório (commitar após rodar `train_model.py`).

---

## 📋 Perguntas Respondidas

| # | Indicador | Principal Finding |
|---|-----------|-------------------|
| 1 | IAN — perfil de defasagem | 55.7% dos alunos apresentam algum nível de defasagem; casos severos são raros (1.5%) |
| 2 | IDA — tendência de desempenho | Desempenho tende a crescer nas fases iniciais mas oscila nas avançadas |
| 3 | IEG × IDA × IPV | Forte correlação positiva: engajamento é preditor robusto de desempenho |
| 4 | IAA vs desempenho real | 71.8% dos alunos superestimam seu desempenho; delta médio de 1.55 pontos |
| 5 | IPS e quedas futuras | IPS baixo antecede quedas de IDA no ano seguinte — sinal de alerta precoce |
| 6 | IPP × IAN | IPP é consistente com IAN: alunos defasados têm IPP significativamente menor |
| 7 | Drivers do IPV | IPP (49.5%), IEG (18.8%) e IDA (13.6%) são os maiores drivers do ponto de virada |
| 8 | Multidimensionalidade → INDE | IDA (0.77), IEG (0.71) e IPV (0.68) têm maior correlação com o INDE |
| 9 | Modelo preditivo de risco | Random Forest AUC 0.97 — identifica 87% dos alunos em risco com 95% de acurácia |
| 10 | Efetividade do programa | 24.3% subiram de Pedra; Topázio cresceu de 15% para 31% entre 2022 e 2024 |
| 11 | Insights criativos | Fase 9 tem 100% de risco; meninas têm INDE ligeiramente maior; fases iniciais mais vulneráveis |

---

## 👥 Equipe

Desenvolvido para o **Datathon POSTECH — Fase 5**

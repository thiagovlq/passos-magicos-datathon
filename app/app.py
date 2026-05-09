"""
app.py — Streamlit App: Preditor de Risco de Defasagem
Passos Mágicos — Datathon POSTECH Fase 5

Deploy: Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuração da página ──────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Preditor de Risco",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilo CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e6da4 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #cce0f5; margin: 0.3rem 0 0; font-size: 1rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid;
    }
    .card-verde  { border-color: #27ae60; }
    .card-laranja{ border-color: #f39c12; }
    .card-vermelho{ border-color: #e74c3c; }

    .risco-baixo  { background: #d5f5e3; border-radius: 8px; padding: 1rem; border-left: 5px solid #27ae60; color: #1a4a2e; }
    .risco-medio  { background: #fef9e7; border-radius: 8px; padding: 1rem; border-left: 5px solid #f39c12; color: #4a3500; }
    .risco-alto   { background: #fdedec; border-radius: 8px; padding: 1rem; border-left: 5px solid #e74c3c; color: #4a1010; }

    .stButton > button {
        background: linear-gradient(135deg, #1a3a5c, #2e6da4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2e6da4, #1a3a5c);
    }
</style>
""", unsafe_allow_html=True)

# ── Carregamento do modelo ──────────────────────────────────────────
@st.cache_resource
def load_model():
    """Carrega o modelo treinado do disco."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path    = os.path.join(base_dir, '..', 'models', 'random_forest_risco.pkl')
    features_path = os.path.join(base_dir, '..', 'models', 'features.pkl')

    if not os.path.exists(model_path):
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    return model, features

model, ALL_FEATURES = load_model()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Passos Mágicos — Preditor de Risco de Defasagem</h1>
    <p>Modelo preditivo baseado em Random Forest · POSTECH Datathon Fase 5</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("""
    ⚠️ **Modelo não encontrado!**

    Execute o script de treinamento primeiro:
    ```bash
    cd passos_magicos
    python models/train_model.py
    ```
    Depois reinicie o app.
    """)
    st.stop()

# ── Sidebar — Navegação ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px;'>
        <h2 style='color:#2e6da4;'>🎓 Passos Mágicos</h2>
        <p style='color:gray; font-size:12px;'>Transformando vidas pela educação</p>
    </div>
""", unsafe_allow_html=True)
             
    st.markdown("---")
    pagina = st.radio(
        "📌 Navegação",
        ["🔍 Predição Individual", "📊 Análise em Lote", "ℹ️ Sobre o Modelo"],
        index=0
    )
    st.markdown("---")
    st.markdown("**🎓 Datathon POSTECH — Fase 5**")
    st.markdown("Modelo: Random Forest (300 árvores)")
    st.markdown("Features: 16 indicadores + engineered")

# ════════════════════════════════════════════════════════════════════
# PÁGINA 1 — PREDIÇÃO INDIVIDUAL
# ════════════════════════════════════════════════════════════════════
if pagina == "🔍 Predição Individual":

    st.subheader("🔍 Avaliação Individual do Aluno")
    st.markdown("Preencha os indicadores do aluno para calcular a probabilidade de risco de defasagem.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### 📋 Indicadores do Aluno")

        with st.form("form_aluno"):
            c1, c2 = st.columns(2)

            with c1:
                fase = st.selectbox("Fase", options=list(range(9)),
                                    format_func=lambda x: f"Fase {x}" if x > 0 else "Fase 0 (Alfa)",
                                    help="Fase atual do aluno no programa")
                inde = st.slider("INDE — Índice Global", 0.0, 10.0, 7.0, 0.1,
                                 help="Índice de Desenvolvimento Educacional")
                ida  = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 6.5, 0.1)
                ieg  = st.slider("IEG — Engajamento", 0.0, 10.0, 7.5, 0.1)

            with c2:
                iaa  = st.slider("IAA — Autoavaliação", 0.0, 10.0, 8.0, 0.1)
                ips  = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.5, 0.1)
                ipv  = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 7.0, 0.1)
                ian_opcoes = {2.5: "2.5 — Severo (≥2 fases)", 5.0: "5.0 — Moderado (1 fase)", 10.0: "10.0 — Adequado"}
                ian_val = st.selectbox("IAN — Adequação do Nível", options=[2.5, 5.0, 10.0],
                                       format_func=lambda x: ian_opcoes[x])

            ipp = st.slider("IPP — Psicopedagógico (opcional)", 0.0, 10.0, 7.5, 0.1,
                            help="Deixe em 0 se não disponível — será imputado automaticamente")
            ipp_val = ipp if ipp > 0 else np.nan

            submitted = st.form_submit_button("🎯 Calcular Risco", use_container_width=True)

    with col_result:
        if submitted:
            # Feature engineering
            fe_ieg_ida    = ieg * ida
            fe_delta      = iaa - ida
            fe_ips_baixo  = float(ips < 5.0)
            fe_ieg_baixo  = float(ieg < 5.0)
            fe_ida_baixo  = float(ida < 5.0)
            fe_tres       = fe_ips_baixo + fe_ieg_baixo + fe_ida_baixo
            fe_ian_defas  = float(ian_val < 10.0)

            entrada = {
                'Fase': fase, 'INDE': inde, 'IAA': iaa, 'IEG': ieg,
                'IPS': ips, 'IDA': ida, 'IPV': ipv, 'IAN': ian_val, 'IPP': ipp_val,
                'FE_IEG_x_IDA': fe_ieg_ida, 'FE_Delta_IAA_IDA': fe_delta,
                'FE_IPS_baixo': fe_ips_baixo, 'FE_IEG_baixo': fe_ieg_baixo,
                'FE_IDA_baixo': fe_ida_baixo, 'FE_Tres_baixos': fe_tres,
                'FE_IAN_defas': fe_ian_defas
            }

            X_novo = pd.DataFrame([entrada])[ALL_FEATURES]
            prob   = model.predict_proba(X_novo)[0, 1]

            # Classificação de risco
            if prob < 0.35:
                nivel, cor_css, emoji, recomendacao = (
                    "BAIXO", "risco-baixo", "🟢",
                    "Aluno apresenta indicadores favoráveis. Manter acompanhamento regular."
                )
            elif prob < 0.65:
                nivel, cor_css, emoji, recomendacao = (
                    "MÉDIO", "risco-medio", "🟡",
                    "Atenção necessária. Recomenda-se intervenção preventiva e acompanhamento próximo."
                )
            else:
                nivel, cor_css, emoji, recomendacao = (
                    "ALTO", "risco-alto", "🔴",
                    "Risco elevado de defasagem! Intervenção imediata recomendada pela equipe pedagógica."
                )

            st.markdown("#### 📈 Resultado da Predição")

            # Gauge de probabilidade
            fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=False))
            fig.patch.set_alpha(0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')

            # Barra de fundo
            ax.barh(0.5, 1.0, height=0.3, left=0, color='#ecf0f1', zorder=1)
            # Barra de progresso colorida
            grad_color = '#27ae60' if prob < 0.35 else '#f39c12' if prob < 0.65 else '#e74c3c'
            ax.barh(0.5, prob, height=0.3, left=0, color=grad_color, zorder=2, alpha=0.85)
            # Marcadores de threshold
            ax.axvline(0.35, ymin=0.25, ymax=0.75, color='#f39c12', linewidth=2, zorder=3)
            ax.axvline(0.65, ymin=0.25, ymax=0.75, color='#e74c3c', linewidth=2, zorder=3)
            # Texto
            ax.text(0.5, 0.88, f'{prob:.1%}', ha='center', va='center',
                    fontsize=32, fontweight='bold', color=grad_color, transform=ax.transAxes)
            ax.text(0.5, 0.12, f'Probabilidade de Risco — Nível {nivel}',
                    ha='center', va='center', fontsize=11, color='#555', transform=ax.transAxes)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Card de resultado
            st.markdown(f"""
            <div class="{cor_css}">
                <h3>{emoji} Risco {nivel}</h3>
                <p><strong>Probabilidade estimada:</strong> {prob:.1%}</p>
                <p><b>Recomendação:</b> {recomendacao}</p>
            </div>
            """, unsafe_allow_html=True)

            # Indicadores de alerta
            st.markdown("#### ⚠️ Indicadores de Atenção")
            alertas = []
            if ieg < 5.0:  alertas.append(("IEG baixo", f"{ieg:.1f}", "Engajamento abaixo do esperado"))
            if ida < 5.0:  alertas.append(("IDA baixo", f"{ida:.1f}", "Desempenho acadêmico crítico"))
            if ips < 5.0:  alertas.append(("IPS baixo", f"{ips:.1f}", "Aspecto psicossocial comprometido"))
            if ian_val < 10.0: alertas.append(("IAN defasado", f"{ian_val}", "Aluno fora da fase ideal"))
            if iaa - ida > 2: alertas.append(("Superestimação alta", f"{iaa-ida:.1f}", "IAA muito acima do IDA"))

            if alertas:
                for nome, val, desc in alertas:
                    st.warning(f"**{nome}** ({val}): {desc}")
            else:
                st.success("✅ Nenhum indicador em nível crítico!")
        else:
            st.info("👈 Preencha os indicadores ao lado e clique em **Calcular Risco**.")

            # Card de explicação dos indicadores
            st.markdown("#### 📘 Guia dos Indicadores")
            indicadores_info = {
                "IAN": "Adequação do nível — 2.5 (severo), 5.0 (moderado), 10.0 (adequado)",
                "IDA": "Desempenho acadêmico (0–10)",
                "IEG": "Engajamento nas atividades (0–10)",
                "IAA": "Autoavaliação do aluno (0–10)",
                "IPS": "Aspectos psicossociais (0–10)",
                "IPP": "Avaliação psicopedagógica (0–10)",
                "IPV": "Ponto de virada — potencial de evolução (0–10)",
                "INDE": "Índice global de desenvolvimento (0–10)",
            }
            for sigla, desc in indicadores_info.items():
                st.markdown(f"- **{sigla}**: {desc}")


# ════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ANÁLISE EM LOTE
# ════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análise em Lote":

    st.subheader("📊 Análise em Lote — Upload de Dados")
    st.markdown("Faça upload de um CSV com os indicadores dos alunos para calcular o risco de toda a turma.")

    col_tmpl, col_upload = st.columns(2)

    with col_tmpl:
        st.markdown("**📥 Modelo de CSV esperado:**")
        template = pd.DataFrame({
            'RA'  : ['RA-001', 'RA-002'],
            'Fase': [3, 5],
            'INDE': [6.5, 8.2],
            'IAA' : [7.0, 9.0],
            'IEG' : [5.5, 8.5],
            'IPS' : [6.0, 7.5],
            'IDA' : [5.0, 7.8],
            'IPV' : [6.2, 8.0],
            'IAN' : [5.0, 10.0],
            'IPP' : [7.0, 8.0],
        })
        st.dataframe(template, use_container_width=True)
        csv_template = template.to_csv(index=False)
        st.download_button("⬇️ Baixar template CSV", csv_template,
                           "template_alunos.csv", "text/csv")

    with col_upload:
        uploaded = st.file_uploader("📤 Upload do CSV da turma", type=['csv'])

    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        st.success(f"✅ Arquivo carregado: {len(df_upload)} alunos")

        # Feature engineering
        df_upload['FE_IEG_x_IDA']    = df_upload['IEG'] * df_upload['IDA']
        df_upload['FE_Delta_IAA_IDA']= df_upload['IAA'] - df_upload['IDA']
        df_upload['FE_IPS_baixo']    = (df_upload['IPS'] < 5.0).astype(float)
        df_upload['FE_IEG_baixo']    = (df_upload['IEG'] < 5.0).astype(float)
        df_upload['FE_IDA_baixo']    = (df_upload['IDA'] < 5.0).astype(float)
        df_upload['FE_Tres_baixos']  = (df_upload['FE_IPS_baixo'] +
                                        df_upload['FE_IEG_baixo'] +
                                        df_upload['FE_IDA_baixo'])
        df_upload['FE_IAN_defas']    = (df_upload['IAN'] < 10.0).astype(float)
        if 'IPP' not in df_upload.columns:
            df_upload['IPP'] = np.nan

        X_batch = df_upload[ALL_FEATURES]
        df_upload['Prob_Risco'] = model.predict_proba(X_batch)[:, 1]
        df_upload['Nivel_Risco'] = df_upload['Prob_Risco'].apply(
            lambda p: '🔴 ALTO' if p >= 0.65 else '🟡 MÉDIO' if p >= 0.35 else '🟢 BAIXO'
        )

        # Métricas gerais
        st.markdown("---")
        st.markdown("#### 📊 Resumo da Turma")
        m1, m2, m3, m4 = st.columns(4)
        alto   = (df_upload['Prob_Risco'] >= 0.65).sum()
        medio  = ((df_upload['Prob_Risco'] >= 0.35) & (df_upload['Prob_Risco'] < 0.65)).sum()
        baixo  = (df_upload['Prob_Risco'] < 0.35).sum()
        media  = df_upload['Prob_Risco'].mean()

        m1.metric("🔴 Risco Alto", f"{alto} alunos", f"{alto/len(df_upload)*100:.1f}%")
        m2.metric("🟡 Risco Médio", f"{medio} alunos", f"{medio/len(df_upload)*100:.1f}%")
        m3.metric("🟢 Risco Baixo", f"{baixo} alunos", f"{baixo/len(df_upload)*100:.1f}%")
        m4.metric("📈 Prob. Média", f"{media:.1%}")

        # Tabela de resultados
        cols_show = ['RA'] if 'RA' in df_upload.columns else []
        cols_show += ['Fase', 'IDA', 'IEG', 'IPS', 'IAN', 'INDE', 'Prob_Risco', 'Nivel_Risco']
        cols_show = [c for c in cols_show if c in df_upload.columns]

        st.markdown("#### 📋 Resultados por Aluno")
        df_show = df_upload[cols_show].sort_values('Prob_Risco', ascending=False)
        df_show['Prob_Risco'] = df_show['Prob_Risco'].map('{:.1%}'.format)
        st.dataframe(df_show, use_container_width=True)

        # Download
        csv_result = df_upload.to_csv(index=False)
        st.download_button("⬇️ Baixar resultados", csv_result,
                           "resultados_risco.csv", "text/csv")

        # Gráfico distribuição
        fig, ax = plt.subplots(figsize=(8, 3.5))
        cores = ['#27ae60', '#f39c12', '#e74c3c']
        ax.bar(['🟢 Baixo', '🟡 Médio', '🔴 Alto'], [baixo, medio, alto],
               color=cores, edgecolor='white', linewidth=1.5)
        for x, val, tot in zip(['🟢 Baixo','🟡 Médio','🔴 Alto'], [baixo,medio,alto], [len(df_upload)]*3):
            ax.text(x, val + 0.3, f'{val}\n({val/tot*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
        ax.set_title('Distribuição de Risco na Turma', fontsize=12)
        ax.set_ylabel('Nº de Alunos')
        ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ════════════════════════════════════════════════════════════════════
# PÁGINA 3 — SOBRE O MODELO
# ════════════════════════════════════════════════════════════════════
elif pagina == "ℹ️ Sobre o Modelo":

    st.subheader("ℹ️ Sobre o Modelo Preditivo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🌳 Random Forest — Arquitetura

        | Parâmetro | Valor |
        |-----------|-------|
        | Algoritmo | Random Forest |
        | Árvores | 300 |
        | Profundidade máxima | 10 |
        | Min. amostras por folha | 5 |
        | Balanceamento | `class_weight='balanced'` |
        | Imputação | Mediana (para IPP nulo) |

        #### 🎯 Variável-Alvo
        `EM_RISCO = 1` quando:
        - Pedra classificada como **QUARTZO**, ou
        - **Defasagem > 0** (aluno atrás da fase ideal)
        """)

    with col2:
        st.markdown("""
        #### 📐 Features Utilizadas (16 no total)

        **Indicadores base (9):**
        - Fase, INDE, IAA, IEG, IPS, IDA, IPV, IAN, IPP

        **Features Engineered (7):**
        - `IEG × IDA` — interação engajamento × desempenho
        - `IAA − IDA` — nível de superestimação
        - `IPS_baixo` — flag psicossocial crítico
        - `IEG_baixo` — flag desengajamento
        - `IDA_baixo` — flag desempenho crítico
        - `Tres_baixos` — acúmulo de flags críticos
        - `IAN_defas` — flag qualquer defasagem

        #### 📊 Performance (conjunto de teste)
        - **AUC-ROC**: 0.9733 (teste) · 0.9762 ± 0.0078 (CV 5-fold)
        - **Recall** (alunos em risco): 87%
        - **Acurácia geral**: 95%
        - **Cross-validation** 5-fold estratificado
        """)

    st.markdown("---")
    st.markdown("""
    #### 🏢 Sobre a Passos Mágicos

    A **Associação Passos Mágicos** é uma ONG que transforma a vida de crianças e jovens em situação de
    vulnerabilidade social através da educação de qualidade, apoio psicossocial e orientação profissional.

    Este modelo preditivo foi desenvolvido no âmbito do **Datathon POSTECH — Fase 5** para auxiliar
    a equipe pedagógica a identificar precocemente alunos em risco de defasagem educacional.

    > *"A educação é a arma mais poderosa que você pode usar para mudar o mundo."* — Nelson Mandela
    """)

    st.markdown("---")
    st.markdown("""
    **Desenvolvido por:** Grupo Datathon POSTECH
    **Framework:** Streamlit · scikit-learn · pandas · matplotlib
    """)

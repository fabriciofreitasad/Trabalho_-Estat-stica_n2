import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_ind, zscore

# Carregar os dados com tratamento
@st.cache_data
def carregar_dados():
    df = pd.read_csv("importacoes-exportacoes.csv", sep=";")

    # Converter vírgula para ponto e transformar em float
    df["IMPORTADO / EXPORTADO"] = df["IMPORTADO / EXPORTADO"].str.replace(",", ".").astype(float)

    # Remover duplicatas
    df = df.drop_duplicates()

    # Remover valores ausentes
    df = df.dropna()

    # Remover registros com volume igual a zero (dados não informativos)
    df = df[df["IMPORTADO / EXPORTADO"] > 0]

    return df

df = carregar_dados()

# Etapas interativas
with st.sidebar:
    st.title("Estudo ANP")
    etapa = st.radio("Selecione a etapa:", [
        "1. Limpeza", "2. Quartis e Outliers", "3. Z-score",
        "4. Intervalo de Confiança", "5. Testes de Hipótese"])

st.title("Análise Estatística - Derivados de Petróleo")

if etapa == "1. Limpeza":
    st.subheader("Etapa 1 - Limpeza de Dados")
    st.markdown("""
    - Foram removidas duplicatas.
    - Valores nulos foram descartados.
    - Volumes com valor **zero** foram eliminados.
    - Volume convertido de texto para **float** com separador decimal.
    """)
    st.write(f"Total de registros válidos: {len(df)}")
    st.write(f"Período: {df['ANO'].min()} a {df['ANO'].max()}")
    st.write(df.describe())

elif etapa == "2. Quartis e Outliers":
    st.subheader("Etapa 2 - Quartis e Outliers")
    q1 = df["IMPORTADO / EXPORTADO"].quantile(0.25)
    q2 = df["IMPORTADO / EXPORTADO"].quantile(0.50)
    q3 = df["IMPORTADO / EXPORTADO"].quantile(0.75)
    iqr = q3 - q1
    limite_sup = q3 + 1.5 * iqr
    outliers = df[df["IMPORTADO / EXPORTADO"] > limite_sup]

    st.write(f"Q1: {q1:.2f}, Q2 (Mediana): {q2:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    st.write(f"Outliers (acima de Q3 + 1.5xIQR): {len(outliers)}")

    fig, ax = plt.subplots()
    sns.boxplot(x=df["IMPORTADO / EXPORTADO"], ax=ax)
    ax.set_title("Boxplot do Volume")
    st.pyplot(fig)

elif etapa == "3. Z-score":
    st.subheader("Etapa 3 - Z-score")
    df["Z"] = zscore(df["IMPORTADO / EXPORTADO"])
    acima_1 = df[df["Z"] > 1]
    st.write(f"Valores com Z > 1: {len(acima_1)}")
    st.write(f"Probabilidade Z > 1.96: {1 - norm.cdf(1.96):.2%}")

    fig, ax = plt.subplots()
    sns.histplot(df["Z"], bins=50, kde=True, stat="density", ax=ax)
    x = np.linspace(-4, 4, 100)
    ax.plot(x, norm.pdf(x), 'r--', label="Normal")
    ax.legend()
    ax.set_title("Distribuição dos Z-scores")
    st.pyplot(fig)

elif etapa == "4. Intervalo de Confiança":
    st.subheader("Etapa 4 - Intervalo de Confiança")
    amostra = df.sample(300, random_state=42)
    z = 1.96

    media_v = amostra["IMPORTADO / EXPORTADO"].mean()
    std_v = amostra["IMPORTADO / EXPORTADO"].std()
    erro_v = z * (std_v / np.sqrt(300))

    media_d = amostra["DISPÊNDIO / RECEITA"].mean()
    std_d = amostra["DISPÊNDIO / RECEITA"].std()
    erro_d = z * (std_d / np.sqrt(300))

    st.write(f"Volume: {media_v:.2f} ± {erro_v:.2f}")
    st.write(f"Dispêndio: {media_d:.2f} ± {erro_d:.2f}")

elif etapa == "5. Testes de Hipótese":
    st.subheader("Etapa 5 - Testes de Hipótese")
    gasolina = df[(df["PRODUTO"] == "GASOLINA A") & (df["OPERAÇÃO COMERCIAL"] == "EXPORTAÇÃO")]
    diesel = df[(df["PRODUTO"] == "ÓLEO DIESEL") & (df["OPERAÇÃO COMERCIAL"] == "EXPORTAÇÃO")]
    t_stat, p_val = ttest_ind(gasolina["IMPORTADO / EXPORTADO"], diesel["IMPORTADO / EXPORTADO"], equal_var=False)

    st.write("Hipótese 1 - Volume Gasolina A x Óleo Diesel (Exportação)")
    st.write(f"t = {t_stat:.2f}, p = {p_val:.4e}")
    st.write("Resultado:", "Rejeita H0" if p_val < 0.05 else "Não rejeita H0")

    st.divider()

    prop = (df["DISPÊNDIO / RECEITA"] > 500_000).mean()
    z_score = (prop - 0.5) / np.sqrt(0.5 * 0.5 / len(df))
    p_prop = 2 * (1 - norm.cdf(abs(z_score)))

    st.write("Hipótese 2 - Proporção dispêndio > 500 mil vs 0.5")
    st.write(f"Proporção observada: {prop:.2%}")
    st.write(f"z = {z_score:.2f}, p = {p_prop:.4e}")
    st.write("Resultado:", "Rejeita H0" if p_prop < 0.05 else "Não rejeita H0")

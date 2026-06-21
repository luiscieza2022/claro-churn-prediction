import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard General", page_icon="📊", layout="wide")
st.title("📊 Análisis Exploratorio de Clientes")

@st.cache_data
def load_data():
    return pd.read_excel("data/Telco_customer_churn.xlsx")

try:
    df = load_data()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Clientes", f"{len(df):,}")
    col2.metric("Tasa de Abandono Global", f"{(df['Churn Value'].mean() * 100):.1f}%")
    col3.metric("Ingreso Mensual Promedio", f"S/. {df['Monthly Charges'].mean():.2f}")

    st.write("---")
    st.subheader("Distribución de Abandono por Tipo de Contrato")
    
    churn_contract = df.groupby(['Contract', 'Churn Label']).size().reset_index(name='Count')
    fig = px.bar(churn_contract, x="Contract", y="Count", color="Churn Label", 
                 barmode="group", color_discrete_sequence=["#DA291C", "#1F77B4"])
    st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("No se encontró el dataset en la ruta 'data/Telco_customer_churn.xlsx'.")
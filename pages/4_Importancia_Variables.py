import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

st.set_page_config(page_title="Importancia de Variables", page_icon="📈", layout="wide")
st.title("📈 Jerarquización de Factores de Abandono")
st.markdown("""
Identificación de variables que tienen mayor impacto en la decisión de un cliente de abandonar el servicio de Claro.
""")

@st.cache_resource
def load_model_data():
    with open("models/modelo_logistica.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/features_list.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

try:
    model, features_list = load_model_data()
    
    coeficientes = model.coef_[0]
    
    df_importancia = pd.DataFrame({
        'Variable': features_list,
        'Impacto': coeficientes
    })
    
    df_importancia['Fuerza_Absoluta'] = df_importancia['Impacto'].abs()
    
    df_importancia['Tipo de Impacto'] = df_importancia['Impacto'].apply(
        lambda x: 'Aumenta el Riesgo (Fuga)' if x > 0 else 'Aumenta la Retención (Fidelidad)'
    )
    
    df_importancia = df_importancia.sort_values(by='Fuerza_Absoluta', ascending=True).tail(15)
    
    st.write("---")
    st.subheader("Top 15 Variables más Críticas para el Modelo")
    
    fig = px.bar(
        df_importancia, 
        x="Impacto", 
        y="Variable", 
        color="Tipo de Impacto",
        orientation='h',
        color_discrete_map={
            'Aumenta el Riesgo (Fuga)': '#DA291C',
            'Aumenta la Retención (Fidelidad)': '#1F77B4'
        },
        title="Impacto de cada variable en la probabilidad matemática de Churn"
    )
    
    fig.update_layout(height=600, yaxis_title="Factores y Características del Cliente", xaxis_title="Peso Logístico (Coeficiente Beta)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Interpretación para Marketing:** Las barras rojas hacia la derecha indican que poseer esa característica dispara la probabilidad de abandono (ej. Fibra óptica de mala calidad). Las barras azules hacia la izquierda indican que la característica hace que el cliente se quede en Claro (ej. Mayor antigüedad o contratos largos).")

except FileNotFoundError:
    st.error("No se encontró el modelo. Ejecuta el entrenamiento primero.")
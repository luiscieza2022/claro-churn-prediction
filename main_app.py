import streamlit as st

st.set_page_config(
    page_title="Sistema de Retención Claro",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #DA291C; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #333333; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Sistema Inteligente de Predicción de Abandono (Churn)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">América Móvil Perú S.A.C (Claro) 🔴</p>', unsafe_allow_html=True)

st.write("---")

st.markdown("""
### 📌 Objetivo del Sistema
Esta plataforma implementa modelos de **Aprendizaje Estadístico** para predecir la probabilidad de que un cliente de servicios fijos y móviles decida abandonar la compañía (Churn). 
A través del análisis de variables críticas como la antigüedad, tipo de contrato y servicios contratados, el sistema permite al área de retención y marketing tomar decisiones proactivas.

### 👥 Equipo de Desarrollo (Ingeniería de Sistemas e Inteligencia Artificial)
* Amador Contreras, Xiomara Yamile
* Barba Ponce, Mauricio Jesus
* Carranza Ortiz, Alejandro Gabriel
* Cieza Llanos, Luis Eduardo
* Roque Saavedra, Jean Pierre

**Docente:** Sagastegui Chigne, Teobaldo  
**Año:** 2026
""")

st.info("👈 Selecciona una opción en el menú lateral para explorar el Dashboard Analítico o el Simulador de Riesgo.")
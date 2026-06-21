import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Simulador de Segmentos", page_icon="🔮", layout="wide")
st.title("🔮 Simulador Avanzado por Segmentación y Búsqueda Directa")
st.markdown("""
Filtre la base de datos de Claro según variables de comportamiento o ingrese un **CustomerID** específico 
para evaluar el riesgo predictivo en tiempo real y desplegar planes de acción tempranos.
""")

@st.cache_resource
def load_models():
    with open("models/modelo_logistica.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/features_list.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

@st.cache_data
def load_raw_data():
    return pd.read_excel("data/Telco_customer_churn.xlsx")

try:
    model, scaler, features_list = load_models()
    df_raw = load_raw_data()
except Exception as e:
    st.error("Error cargando archivos base. Ejecuta primero 'src/model_trainer.py'.")
    st.stop()

st.write("---")
st.subheader("🛠️ Criterios de Selección (Búsqueda Individual o Filtrado por Segmento)")

# Caja de búsqueda individual (Mundo Módulo 5)
search_id = st.text_input("🔍 Búsqueda Directa por ID (Opcional): Ingrese el CustomerID para aislar un cliente", placeholder="Ejemplo: 3668-QPYBK")

st.markdown("**O filtre por características grupales (se ignorará si escribe un ID arriba):**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    f_contract = st.selectbox("Tipo de Contrato", ["Todos", "Month-to-month", "One year", "Two year"])
with col2:
    f_internet = st.selectbox("Servicio de Internet", ["Todos", "Fiber optic", "DSL", "No"])
with col3:
    f_payment = st.selectbox("Método de Pago", ["Todos", "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
with col4:
    f_dependents = st.selectbox("¿Tiene Dependientes?", ["Todos", "Yes", "No"])

df_filtered = df_raw.copy()

if search_id:
    search_id = search_id.strip()
    df_filtered = df_filtered[df_filtered['CustomerID'] == search_id]
    is_individual = True
else:
    if f_contract != "Todos":
        df_filtered = df_filtered[df_filtered['Contract'] == f_contract]
    if f_internet != "Todos":
        df_filtered = df_filtered[df_filtered['Internet Service'] == f_internet]
    if f_payment != "Todos":
        df_filtered = df_filtered[df_filtered['Payment Method'] == f_payment]
    if f_dependents != "Todos":
        df_filtered = df_filtered[df_filtered['Dependents'] == f_dependents]
    is_individual = False

# --- PROCESAMIENTO Y EJECUCIÓN ---
total_encontrados = len(df_filtered)

if total_encontrados == 0:
    if search_id:
        st.warning(f"⚠️ No se encontró ningún cliente con el CustomerID: '{search_id}'. Verifique el código.")
    else:
        st.warning("⚠️ No se encontraron clientes que coincidan con la combinación de filtros seleccionada.")
else:
    if is_individual:
        st.success(f"👤 Cliente '{search_id}' localizado en la base de datos.")
    else:
        st.info(f"📊 Se localizaron {total_encontrados} clientes que pertenecen a este segmento específico.")
    
    if st.button("🚀 Ejecutar Motor de Diagnóstico Predictivo", type="primary", use_container_width=True):
        with st.spinner('Procesando datos con la Regresión Logística...'):
            
            # Formatear el set de datos filtrado
            df_input = df_filtered.copy()
            df_input['Total Charges'] = pd.to_numeric(df_input['Total Charges'], errors='coerce').fillna(df_input['Total Charges'].median())
            
            # Aplicar One-Hot Encoding
            X_encoded = pd.get_dummies(df_input, drop_first=True)
            
            # Alinear columnas con el modelo
            X_aligned = pd.DataFrame(0, index=np.arange(len(X_encoded)), columns=features_list)
            for col in features_list:
                if col in X_encoded.columns:
                    X_aligned[col] = X_encoded[col].values
            
            # Normalizar
            columnas_continuas = ['Tenure Months', 'Monthly Charges', 'Total Charges']
            X_aligned[columnas_continuas] = scaler.transform(X_aligned[columnas_continuas])
            
            # Inferencia
            probabilidades = model.predict_proba(X_aligned)[:, 1]
            predicciones = model.predict(X_aligned)
            
            # Añadir métricas calculadas
            df_display = df_filtered.copy()
            df_display.insert(0, 'Probabilidad Abandono (%)', (probabilidades * 100).round(2))
            df_display.insert(1, 'Estado de Riesgo', predicciones)
            df_display['Estado de Riesgo'] = df_display['Estado de Riesgo'].map({1: '🔴 ALTO RIESGO', 0: '✅ Estable'})
            
            # --- SECCIÓN DE MÉTRICAS ---
            st.write("---")
            if is_individual:
                st.subheader("📈 Resultado del Análisis de Cliente Individual")
                prob_ind = probabilidades[0] * 100
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Probabilidad de Fuga Directa", f"{prob_ind:.2f}%")
                if prob_ind >= 50:
                    m_col2.error("🚨 CRÍTICO: CLIENTE CON ALTO RIESGO DE ABANDONO")
                elif prob_ind >= 30:
                    m_col2.warning("⚠️ ADVERTENCIA: RIESGO MODERADO")
                else:
                    m_col2.success("✅ ESTABLE: BAJA PROBABILIDAD DE FUGA")
            else:
                st.subheader("📈 Diagnóstico General del Segmento")
                m_col1, m_col2, m_col3 = st.columns(3)
                riesgo_promedio = probabilidades.mean() * 100
                clientes_criticos = len(df_display[df_display['Estado de Riesgo'] == '🔴 ALTO RIESGO'])
                
                m_col1.metric("Riesgo Promedio del Grupo", f"{riesgo_promedio:.2f}%")
                m_col2.metric("Clientes en Alerta Roja", f"{clientes_criticos} usuarios")
                m_col3.metric("Facturación Mensual Expuesta", f"S/. {df_filtered['Monthly Charges'].sum():,.2f}")
            
            # --- PLAN DE ACCIÓN EXPLICATIVO ---
            st.write("---")
            st.subheader("📋 Plan de Acción y Alertas del Sistema")
            
            causas, soluciones = [], []
            
            # Extraer variables dinámicamente según sea un caso individual o grupal
            val_contract = df_filtered['Contract'].values[0] if is_individual else f_contract
            val_internet = df_filtered['Internet Service'].values[0] if is_individual else f_internet
            val_payment = df_filtered['Payment Method'].values[0] if is_individual else f_payment
            
            if val_contract == "Month-to-month" or (not is_individual and f_contract == "Todos"):
                causas.append("Contratación flexible (mes a mes) que reduce las barreras de salida y facilita la fuga hacia competidores.")
                soluciones.append("Ofrecer un descuento exclusivo de retención (ej. 15%) condicionado a una migración voluntaria hacia un contrato anual.")
                
            if val_internet == "Fiber optic" or (not is_individual and f_internet == "Todos"):
                causas.append("Rasgo estructural de insatisfacción técnica o sensibilidad al costo en la infraestructura de Fibra Óptica.")
                soluciones.append("Priorizar auditoría técnica remota de estabilidad de línea o incluir asistencia técnica especializada sin costo.")
                
            if val_payment == "Electronic check" or (not is_individual and f_payment == "Todos"):
                causas.append("Los métodos de cobro manuales o digitales interactivos registran picos históricos de abandono involuntario.")
                soluciones.append("Desplegar beneficio promocional (Claro Puntos o GB adicionales) si el usuario afilia su cuenta al Débito Automático.")
            
            if is_individual and df_filtered['Tenure Months'].values[0] < 12:
                causas.append("Cliente en periodo crítico de adaptación (menos de 1 año). Muy vulnerable frente a ofertas agresivas externas.")
                soluciones.append("Vincular inmediatamente al programa Claro Club y agendar contacto de cortesía post-venta.")

            if not causas:
                causas.append("No se detectan anomalías extremas en los parámetros contractuales clásicos.")
                soluciones.append("Mantener el protocolo regular de comunicación y aplicar campañas estándar de fidelización.")

            strat_col1, strat_col2 = st.columns(2)
            with strat_col1:
                st.markdown("### ⚠️ Factores Críticos Identificados")
                for c in causas:
                    st.markdown(f"* {c}")
            with strat_col2:
                st.markdown("### 🛡️ Intervención Temprana Sugerida")
                for s in soluciones:
                    st.markdown(f"* {s}")
            
            # --- TABLA DETALLADA ---
            st.write("---")
            st.subheader("👥 Desglose de Registros Evaluados")
            columnas_visibles = ['CustomerID', 'Probabilidad Abandono (%)', 'Estado de Riesgo', 'Tenure Months', 'Contract', 'Internet Service', 'Payment Method', 'Monthly Charges']
            st.dataframe(df_display[columnas_visibles].sort_values(by='Probabilidad Abandono (%)', ascending=False), use_container_width=True)
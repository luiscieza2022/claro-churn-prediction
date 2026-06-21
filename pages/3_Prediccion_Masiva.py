import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Predicción Masiva", page_icon="📂", layout="wide")
st.title("Predicción Masiva de Churn para Clientes")
st.markdown("Sube una base de datos para evaluar el riesgo de fuga de miles de clientes simultáneamente.")

@st.cache_resource
def load_models():
    with open("models/modelo_logistica.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/features_list.pkl", "rb") as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, features_list = load_models()
except FileNotFoundError:
    st.error("Faltan los archivos del modelo. Ejecuta primero 'src/model_trainer.py'.")
    st.stop()

uploaded_file = st.file_uploader("Sube tu archivo Excel o CSV de clientes a evaluar", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
            
        st.success("Archivo cargado y leído correctamente.")
        
        if st.button("Ejecutar Análisis Masivo de Churn", type="primary"):
            with st.spinner('Procesando miles de registros a través de la Regresión Logística...'):
                
                df_results = df_input.copy()
                
                if 'Total Charges' in df_input.columns:
                    df_input['Total Charges'] = pd.to_numeric(df_input['Total Charges'], errors='coerce')
                    df_input['Total Charges'] = df_input['Total Charges'].fillna(df_input['Total Charges'].median())
                
                X_encoded = pd.get_dummies(df_input, drop_first=True)
                
                
                X_aligned = pd.DataFrame(0, index=np.arange(len(X_encoded)), columns=features_list)
                
                for col in features_list:
                    if col in X_encoded.columns:
                        X_aligned[col] = X_encoded[col].values
                        
                columnas_continuas = ['Tenure Months', 'Monthly Charges', 'Total Charges']
                X_aligned[columnas_continuas] = scaler.transform(X_aligned[columnas_continuas])
                
                probabilidades = model.predict_proba(X_aligned)[:, 1]
                predicciones = model.predict(X_aligned)
                
                df_results.insert(0, 'Alerta_Churn', predicciones)
                df_results.insert(1, 'Probabilidad_Fuga (%)', (probabilidades * 100).round(2))
                
                df_results['Alerta_Churn'] = df_results['Alerta_Churn'].map({1: 'ALTO RIESGO', 0: 'Retenido'})
                
                st.write("---")
                st.subheader("📊 Resultados de la Campaña de Retención")
                
                clientes_riesgo = len(df_results[df_results['Alerta_Churn'] == 'ALTO RIESGO'])
                st.metric(label="Total de Clientes en Peligro Inminente", value=clientes_riesgo)
                
                st.dataframe(df_results.sort_values(by='Probabilidad_Fuga (%)', ascending=False))
                
                # 8. Generar archivo descargable
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Base de Datos con Predicciones (CSV)",
                    data=csv,
                    file_name='Claro_Reporte_Prediccion_Masiva.csv',
                    mime='text/csv',
                )
                
    except Exception as e:
        st.error(f"Ocurrió un error en el procesamiento: {e}")
        st.info("Asegúrate de que el archivo subido tenga las columnas base del sistema.")
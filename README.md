# Sistema Inteligente de Predicción de Abandono (Churn) - Claro
**Curso:** Aprendizaje Estadístico | **Universidad Privada Antenor Orrego (UPAO)** - 2026

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)

## Descripción del Proyecto
Este repositorio contiene el código fuente y la aplicación web interactiva para la predicción de abandono de clientes (Churn) de la empresa de telecomunicaciones **América Móvil Perú S.A.C (Claro)**.

El sistema fue desarrollado utilizando un modelo de **Regresión Logística** (Accuracy: ~81%, AUC-ROC: ~0.85) entrenado con datos históricos, permitiendo al área de retención y marketing evaluar el riesgo predictivo en tiempo real y desplegar planes de acción oportunos.

## Equipo de Desarrollo
* Amador Contreras, Xiomara Yamile
* Barba Ponce, Mauricio Jesus
* Carranza Ortiz, Alejandro Gabriel
* Cieza Llanos, Luis Eduardo
* Roque Saavedra, Jean Pierre

**Docente:** Sagastegui Chigne, Teobaldo

## Arquitectura del Sistema
El proyecto está estructurado de manera modular para separar el backend analítico del frontend interactivo:
* `data/`: Contiene el dataset histórico (`Telco_customer_churn.xlsx`).
* `models/`: Almacena los artefactos pre-entrenados del modelo `.pkl` y el escalador de variables.
* `src/`: Scripts de procesamiento de datos (`data_processor.py`) y entrenamiento automatizado (`model_trainer.py`).
* `pages/`: Módulos de la aplicación Streamlit (Dashboard, Simulador de Segmentos, Predicción Masiva e Interpretabilidad).
* `main_app.py`: Archivo principal de ejecución de la interfaz gráfica.

## Guía de Instalación y Ejecución

**1. Clonar el repositorio e ingresar a la carpeta:**
```bash
git clone https://github.com/luiscieza2022/claro-churn-prediction.git
cd claro-churn-prediction

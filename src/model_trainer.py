import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from data_processor import load_and_preprocess_data

def train_and_save():
    print("⏳ Cargando y procesando datos...")
    X, y, features_list = load_and_preprocess_data("../data/Telco_customer_churn.xlsx")
    
    print("⏳ Dividiendo y escalando datos...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    columnas_continuas = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_train_scaled[columnas_continuas] = scaler.fit_transform(X_train[columnas_continuas])
    
    print("⏳ Entrenando el modelo de Regresión Logística...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("⏳ Guardando artefactos en la carpeta 'models/'...")
    os.makedirs("../models", exist_ok=True)
    
    with open("../models/modelo_logistica.pkl", "wb") as f:
        pickle.dump(model, f)
        
    with open("../models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    with open("../models/features_list.pkl", "wb") as f:
        pickle.dump(features_list, f)
        
    print("✅ ¡Entrenamiento completado y modelos exportados con éxito!")

if __name__ == "__main__":
    train_and_save()
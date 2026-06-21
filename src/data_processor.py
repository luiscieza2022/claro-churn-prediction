import pandas as pd

def load_and_preprocess_data(filepath="data/Telco_customer_churn.xlsx"):
    """
    Carga el dataset, limpia valores nulos y aplica One-Hot Encoding.
    Retorna las variables predictoras (X), la variable objetivo (y) y la lista de columnas.
    """
    df = pd.read_excel(filepath)
    
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())
    
    y = df['Churn Value']
    
    columnas_a_eliminar = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Value',
        'Churn Score', 'CLTV', 'Churn Reason'
    ]
    X = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    features_list = X_encoded.columns.tolist()
    
    return X_encoded, y, features_list
"""
═══════════════════════════════════════════════════════════════════════════════
GUARDAR MODELO ENTRENADO
═══════════════════════════════════════════════════════════════════════════════
Este script entrena y guarda el mejor modelo (Gradient Boosting + SMOTETomek)
para usarlo en la aplicación de predicción.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek

def main():
    print("=" * 60)
    print("📦 GUARDANDO MEJOR MODELO PARA PREDICCIÓN")
    print("=" * 60)
    
    # Crear directorio models/
    os.makedirs('models', exist_ok=True)
    
    # 1. Cargar datos
    print("\n🔄 Cargando dataset...")
    df = pd.read_csv('TACNA_Final_Corregido.csv')
    
    # 2. Preprocesar
    df_clean = df[df['Dx_anemia'].notna()].copy()
    df_clean['anemia_binary'] = df_clean['Dx_anemia'].apply(lambda x: 0 if x == 'Normal' else 1)
    
    feature_cols = ['Sexo', 'EdadMeses', 'Peso', 'Talla', 'PTZ', 'ZTE', 'ZPE', 
                    'AlturaREN', 'Suplementacion', 'SIS']
    available = [c for c in feature_cols if c in df_clean.columns]
    
    df_model = df_clean[available + ['anemia_binary']].copy()
    df_model = df_model.dropna(subset=['EdadMeses', 'Peso', 'Talla'])
    
    # Imputar
    for col in ['PTZ', 'ZTE', 'ZPE', 'AlturaREN']:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
    
    for col in ['Suplementacion', 'SIS']:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
    
    if 'Sexo' in df_model.columns:
        df_model['Sexo'] = df_model['Sexo'].map({'M': 1, 'F': 0}).fillna(0)
    
    # 3. División y escalado
    X = df_model[available]
    y = df_model['anemia_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Aplicar SMOTETomek
    print("🔄 Aplicando SMOTETomek...")
    smotetomek = SMOTETomek(random_state=42)
    X_balanced, y_balanced = smotetomek.fit_resample(X_train_scaled, y_train)
    
    # 5. Entrenar mejor modelo (Gradient Boosting con mejores params)
    print("🔄 Entrenando Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42
    )
    model.fit(X_balanced, y_balanced)
    
    # 6. Guardar modelo y scaler
    print("\n💾 Guardando archivos...")
    
    joblib.dump(model, 'models/best_model.joblib')
    print("   ✅ models/best_model.joblib")
    
    joblib.dump(scaler, 'models/scaler.joblib')
    print("   ✅ models/scaler.joblib")
    
    # 7. Guardar metadata
    model_info = {
        'model_name': 'Gradient Boosting',
        'balancing_technique': 'SMOTETomek',
        'features': available,
        'hyperparameters': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 7
        },
        'training_samples': len(X_balanced),
        'test_samples': len(X_test)
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("   ✅ models/model_info.json")
    
    print("\n" + "=" * 60)
    print("✅ MODELO GUARDADO EXITOSAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()

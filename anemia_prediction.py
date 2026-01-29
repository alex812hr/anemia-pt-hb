"""
═══════════════════════════════════════════════════════════════════════════════
PREDICCIÓN DE ANEMIA CON MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════════════
Dataset: TACNA_Final_Corregido.csv
Modelos: Logistic Regression, Random Forest, Gradient Boosting

TÉCNICAS IMPLEMENTADAS:
- Técnicas de balanceo: SMOTE, Class Weight, Undersampling
- Optimización: GridSearchCV
- Evaluación: F1-Score, Recall, Precision, AUC-ROC

Autor: Generado automáticamente
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTACIÓN DE LIBRERÍAS
# ═══════════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Modelos y métricas
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Imbalanced-learn: Técnicas de balanceo
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import warnings
import os
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN: CARGAR Y EXPLORAR DATOS
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_explore(filepath):
    """
    Carga el dataset y realiza exploración inicial.
    
    Parámetros:
        filepath: Ruta al archivo CSV
    
    Retorna:
        DataFrame con los datos cargados
    """
    print("=" * 70)
    print("📊 CARGA Y EXPLORACIÓN DE DATOS")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"\n✅ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Distribución del target
    print(f"\n📌 Distribución de Dx_anemia:")
    print(df['Dx_anemia'].value_counts(dropna=False))
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN: PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess(df):
    """
    Preprocesa los datos para modelado:
    - Filtra registros válidos
    - Crea target binario
    - Imputa valores faltantes
    - Codifica variables categóricas
    
    Retorna:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("\n" + "=" * 70)
    print("🔧 PREPROCESAMIENTO DE DATOS")
    print("=" * 70)
    
    # 1. Filtrar registros con diagnóstico válido
    df_clean = df[df['Dx_anemia'].notna()].copy()
    
    # 2. Crear target binario (0: Normal, 1: Anemia)
    df_clean['anemia_binary'] = df_clean['Dx_anemia'].apply(
        lambda x: 0 if x == 'Normal' else 1
    )
    
    # 3. Seleccionar features
    feature_cols = ['Sexo', 'EdadMeses', 'Peso', 'Talla', 'PTZ', 'ZTE', 'ZPE', 
                    'AlturaREN', 'Suplementacion', 'SIS']
    available = [c for c in feature_cols if c in df_clean.columns]
    
    # 4. Preparar dataset
    df_model = df_clean[available + ['anemia_binary']].copy()
    df_model = df_model.dropna(subset=['EdadMeses', 'Peso', 'Talla'])
    
    # 5. Imputar valores faltantes
    for col in ['PTZ', 'ZTE', 'ZPE', 'AlturaREN']:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
    
    for col in ['Suplementacion', 'SIS']:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
    
    # 6. Codificar Sexo
    if 'Sexo' in df_model.columns:
        df_model['Sexo'] = df_model['Sexo'].map({'M': 1, 'F': 0}).fillna(0)
    
    print(f"\n✅ Dataset procesado: {len(df_model)} registros")
    print(f"   Sin anemia: {(df_model['anemia_binary'] == 0).sum()}")
    print(f"   Con anemia: {(df_model['anemia_binary'] == 1).sum()}")
    
    # 7. División train/test
    X = df_model[available]
    y = df_model['anemia_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8. Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 División: Train={len(X_train)} | Test={len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN: EVALUAR MODELO
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, name, technique):
    """
    Evalúa un modelo y retorna métricas en diccionario.
    
    Métricas calculadas:
    - Accuracy: Proporción de predicciones correctas
    - Precision: TP / (TP + FP) - Evita falsos positivos
    - Recall: TP / (TP + FN) - Evita falsos negativos (importante en salud)
    - F1-Score: Media armónica de Precision y Recall
    - AUC-ROC: Área bajo la curva ROC
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'Modelo': name,
        'Técnica': technique,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN: ENTRENAR CON TODAS LAS TÉCNICAS DE BALANCEO
# ═══════════════════════════════════════════════════════════════════════════════
def train_all_techniques(X_train, X_test, y_train, y_test):
    """
    Entrena modelos con múltiples técnicas de balanceo:
    
    TÉCNICAS DE BALANCEO:
    1. Sin Balanceo: Baseline
    2. SMOTE: Genera muestras sintéticas de la clase minoritaria
    3. Random Undersampling: Reduce la clase mayoritaria
    4. SMOTETomek: SMOTE + limpieza de muestras ruidosas
    5. Class Weight: Penaliza errores en clase minoritaria
    
    Retorna: DataFrame con todos los resultados
    """
    print("\n" + "=" * 70)
    print("🚀 ENTRENAMIENTO CON TÉCNICAS DE BALANCEO")
    print("=" * 70)
    
    all_results = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # TÉCNICAS DE RESAMPLING
    # ─────────────────────────────────────────────────────────────────────────
    techniques = {
        'Sin Balanceo': None,
        'SMOTE': SMOTE(random_state=42),
        'Undersampling': RandomUnderSampler(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42)
    }
    
    for tech_name, sampler in techniques.items():
        print(f"\n🔄 Técnica: {tech_name}")
        
        # Aplicar balanceo (si corresponde)
        if sampler:
            X_bal, y_bal = sampler.fit_resample(X_train, y_train)
        else:
            X_bal, y_bal = X_train, y_train
        
        # Entrenar los 3 modelos
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for model_name, model in models.items():
            model.fit(X_bal, y_bal)
            result = evaluate(model, X_test, y_test, model_name, tech_name)
            all_results.append(result)
            print(f"   ✅ {model_name}: F1={result['F1-Score']:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLASS WEIGHT (No modifica datos, solo pesos durante entrenamiento)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n🔄 Técnica: Class Weight")
    
    cw_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    }
    
    for model_name, model in cw_models.items():
        model.fit(X_train, y_train)
        result = evaluate(model, X_test, y_test, model_name, 'Class Weight')
        all_results.append(result)
        print(f"   ✅ {model_name}: F1={result['F1-Score']:.4f}")
    
    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN: OPTIMIZACIÓN CON GRIDSEARCHCV
# ═══════════════════════════════════════════════════════════════════════════════
def optimize_models(X_train, X_test, y_train, y_test):
    """
    Optimiza hiperparámetros usando GridSearchCV con SMOTE.
    
    GridSearchCV realiza:
    1. Divide datos en K folds (5)
    2. Para cada combinación de parámetros:
       - Entrena en K-1 folds
       - Valida en 1 fold
       - Repite K veces
    3. Selecciona mejor combinación basada en F1-Score
    """
    print("\n" + "=" * 70)
    print("🔧 OPTIMIZACIÓN CON GRIDSEARCHCV")
    print("=" * 70)
    
    # Aplicar SMOTE primero
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    results = []
    best_models = {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Random Forest - Parámetros a optimizar
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 Optimizando Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid={
            'n_estimators': [50, 100, 200],      # Número de árboles
            'max_depth': [5, 10, 15, None],       # Profundidad máxima
            'min_samples_split': [2, 5, 10]       # Muestras mínimas para dividir
        },
        cv=5, scoring='f1', n_jobs=-1
    )
    rf_grid.fit(X_smote, y_smote)
    print(f"   Mejores params: {rf_grid.best_params_}")
    print(f"   Mejor F1 (CV): {rf_grid.best_score_:.4f}")
    best_models['Random Forest'] = rf_grid.best_estimator_
    
    # ─────────────────────────────────────────────────────────────────────────
    # Logistic Regression - Parámetros a optimizar
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 Optimizando Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid={
            'C': [0.01, 0.1, 1, 10],              # Regularización inversa
            'solver': ['lbfgs', 'saga']           # Algoritmo
        },
        cv=5, scoring='f1', n_jobs=-1
    )
    lr_grid.fit(X_smote, y_smote)
    print(f"   Mejores params: {lr_grid.best_params_}")
    print(f"   Mejor F1 (CV): {lr_grid.best_score_:.4f}")
    best_models['Logistic Regression'] = lr_grid.best_estimator_
    
    # ─────────────────────────────────────────────────────────────────────────
    # Gradient Boosting - Parámetros a optimizar
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 Optimizando Gradient Boosting...")
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid={
            'n_estimators': [50, 100],            # Número de árboles
            'learning_rate': [0.05, 0.1, 0.2],    # Tasa de aprendizaje
            'max_depth': [3, 5, 7]                 # Profundidad
        },
        cv=5, scoring='f1', n_jobs=-1
    )
    gb_grid.fit(X_smote, y_smote)
    print(f"   Mejores params: {gb_grid.best_params_}")
    print(f"   Mejor F1 (CV): {gb_grid.best_score_:.4f}")
    best_models['Gradient Boosting'] = gb_grid.best_estimator_
    
    # Evaluar modelos optimizados en test set
    for name, model in best_models.items():
        result = evaluate(model, X_test, y_test, name, 'SMOTE + GridSearch')
        results.append(result)
        print(f"\n✅ {name} Optimizado: F1={result['F1-Score']:.4f}")
    
    return pd.DataFrame(results), best_models


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN: VISUALIZAR RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════
def visualize_results(df_results, output_dir='outputs'):
    """Genera visualizaciones de resultados."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Heatmap de F1-Score
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot = df_results.pivot(index='Modelo', columns='Técnica', values='F1-Score')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax)
    ax.set_title('F1-Score por Modelo y Técnica de Balanceo', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_f1score.png', dpi=150)
    plt.close()
    print(f"\n✅ Gráfico guardado: {output_dir}/heatmap_f1score.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("   PREDICCIÓN DE ANEMIA CON MACHINE LEARNING")
    print("🔬" * 35)
    
    # Crear directorio de outputs
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Cargar datos
    df = load_and_explore('TACNA_Final_Corregido.csv')
    
    # 2. Preprocesar
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    
    # 3. Entrenar con todas las técnicas de balanceo
    df_techniques = train_all_techniques(X_train, X_test, y_train, y_test)
    
    # 4. Optimizar con GridSearchCV
    df_optimized, best_models = optimize_models(X_train, X_test, y_train, y_test)
    
    # 5. Combinar resultados
    df_all = pd.concat([df_techniques, df_optimized], ignore_index=True)
    
    # 6. Visualizar
    visualize_results(df_all)
    
    # 7. Mostrar tabla final
    print("\n" + "=" * 70)
    print("📊 TABLA COMPARATIVA FINAL")
    print("=" * 70)
    print(df_all.sort_values('F1-Score', ascending=False).to_string(index=False))
    
    # 8. Mejor modelo
    best = df_all.loc[df_all['F1-Score'].idxmax()]
    print("\n" + "🏆" * 25)
    print(f"\n🏆 MEJOR COMBINACIÓN: {best['Modelo']} + {best['Técnica']}")
    print(f"   F1-Score:  {best['F1-Score']:.4f}")
    print(f"   Recall:    {best['Recall']:.4f}")
    print(f"   Precision: {best['Precision']:.4f}")
    print(f"   AUC-ROC:   {best['AUC-ROC']:.4f}")
    
    # 9. Guardar resultados
    df_all.to_csv('outputs/resultados_completos.csv', index=False)
    print("\n✅ Resultados guardados: outputs/resultados_completos.csv")
    
    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("=" * 70)

# 📚 Documentación: Predicción de Anemia con Machine Learning

## 📋 Descripción del Proyecto

Sistema de predicción de anemia en niños usando ML. Incluye técnicas de balanceo y optimización.

---

## 🔧 Técnicas Implementadas

### 1. Técnicas de Balanceo

| Técnica                  | Descripción                           | Cuándo Usar                             |
| ------------------------ | ------------------------------------- | --------------------------------------- |
| **Class Weight**         | Penaliza errores en clase minoritaria | Siempre recomendado como primera opción |
| **Random Undersampling** | Reduce clase mayoritaria              | Datasets grandes                        |
| **SMOTE**                | Genera muestras sintéticas            | Datasets pequeños/medianos              |
| **SMOTETomek**           | SMOTE + limpieza de ruido             | Cuando SMOTE genera ruido               |

### 2. GridSearchCV - Optimización de Hiperparámetros

```python
# Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga']
}

# Gradient Boosting
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
```

---

## 📊 Métricas de Evaluación

| Métrica       | Interpretación                                 |
| ------------- | ---------------------------------------------- |
| **Accuracy**  | % predicciones correctas                       |
| **Precision** | De los positivos predichos, cuántos son reales |
| **Recall**    | De los positivos reales, cuántos detectamos    |
| **F1-Score**  | Balance entre Precision y Recall               |
| **AUC-ROC**   | Capacidad de discriminación                    |

---

## 🚀 Uso

### En Google Colab

1. Subir `Prediccion_Anemia_ML.ipynb` a Colab
2. Ejecutar celdas en orden
3. Subir CSV cuando se solicite

### Variables de Entrada

- `Sexo`: M/F
- `EdadMeses`: Edad en meses
- `Peso`, `Talla`: Medidas antropométricas
- `PTZ`, `ZTE`, `ZPE`: Z-scores nutricionales
- `AlturaREN`: Altitud
- `Suplementacion`, `SIS`: Programas (0/1)

---

## 📈 Resultados Esperados

La tabla de resultados mostrará:

- Rendimiento por modelo × técnica de balanceo
- Mejor combinación basada en F1-Score
- Mejora porcentual vs baseline

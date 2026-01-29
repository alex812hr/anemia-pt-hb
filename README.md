# Anemia Prediction System 🔬

Sistema de predicción de anemia en pacientes pediátricos usando Machine Learning.

## 📊 Descripción

Este proyecto implementa un modelo de Machine Learning para predecir anemia en niños basándose en características demográficas, antropométricas y datos de programas sociales.

### Modelo Utilizado

- **Algoritmo:** Gradient Boosting Classifier
- **Técnica de Balanceo:** SMOTETomek
- **F1-Score:** 0.6645
- **AUC-ROC:** 0.7406

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/anemia-pt-hb.git
cd anemia-pt-hb

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
anemia-pt-hb/
├── app.py                    # Aplicación Streamlit
├── anemia_prediction.py      # Script de entrenamiento
├── save_model.py             # Guardar modelo entrenado
├── models/
│   ├── best_model.joblib     # Modelo entrenado
│   ├── scaler.joblib         # Scaler
│   └── model_info.json       # Metadata
├── outputs/                  # Resultados y gráficos
├── DOCUMENTACION.md          # Documentación técnica
└── requirements.txt          # Dependencias
```

## 🖥️ Uso

### 1. Entrenar modelo (opcional)

```bash
python anemia_prediction.py
```

### 2. Guardar modelo

```bash
python save_model.py
```

### 3. Ejecutar aplicación

```bash
streamlit run app.py
```

Abre http://localhost:8501 en tu navegador.

## 📊 Variables de Entrada

| Variable       | Descripción                    |
| -------------- | ------------------------------ |
| Sexo           | Masculino/Femenino             |
| EdadMeses      | Edad en meses (0-60)           |
| Peso           | Peso en kg                     |
| Talla          | Talla en cm                    |
| PTZ, ZTE, ZPE  | Z-scores nutricionales         |
| AlturaREN      | Altitud del lugar              |
| Suplementacion | Recibe suplementación          |
| SIS            | Tiene Seguro Integral de Salud |

## 📈 Resultados

| Modelo            | Técnica       | F1-Score   | Recall |
| ----------------- | ------------- | ---------- | ------ |
| Gradient Boosting | SMOTETomek    | **0.6645** | 0.6699 |
| Gradient Boosting | Undersampling | 0.6635     | 0.6763 |
| Gradient Boosting | SMOTE         | 0.6603     | 0.6635 |

## ⚠️ Disclaimer

Este sistema es una herramienta de apoyo y NO reemplaza el diagnóstico médico profesional. El diagnóstico definitivo de anemia requiere un examen de hemoglobina en sangre.

## 📝 Licencia

MIT License

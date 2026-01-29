"""
═══════════════════════════════════════════════════════════════════════════════
🔬 SISTEMA DE PREDICCIÓN DE ANEMIA
═══════════════════════════════════════════════════════════════════════════════
Aplicación Streamlit para predecir anemia en pacientes pediátricos.
Usa el modelo Gradient Boosting entrenado con SMOTETomek.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE LA PÁGINA
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Predicción de Anemia",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        padding: 1rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .anemia-positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    .anemia-negative {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    .recommendation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #228be6;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Carga el modelo y scaler guardados."""
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/model_info.json', 'r') as f:
        info = json.load(f)
    return model, scaler, info


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════
def predict_anemia(model, scaler, features):
    """Realiza la predicción de anemia."""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability


def get_recommendations(prediction, prob_anemia, edad_meses, suplementacion, sis):
    """
    Genera recomendaciones automáticas basadas en la predicción.
    
    Recomendaciones basadas en:
    - Resultado de la predicción
    - Probabilidad de anemia
    - Edad del paciente
    - Estado de suplementación
    - Cobertura de seguro
    """
    recommendations = []
    
    if prediction == 1:  # Con anemia
        # Recomendaciones principales
        recommendations.append("🩺 **Consultar con profesional de salud** para confirmación diagnóstica mediante hemograma completo")
        
        if prob_anemia > 0.8:
            recommendations.append("⚠️ **ALTA PROBABILIDAD** - Se recomienda evaluación médica urgente")
        
        recommendations.append("💊 **Considerar suplementación con hierro** bajo supervisión médica")
        recommendations.append("🥗 **Dieta rica en hierro**: carnes rojas, legumbres, vegetales de hoja verde, cereales fortificados")
        recommendations.append("🍊 **Consumir vitamina C** junto con alimentos ricos en hierro para mejorar absorción")
        
        if edad_meses < 24:
            recommendations.append("👶 **Lactantes**: Verificar tipo de alimentación y considerar fórmula fortificada")
        
        if suplementacion == 0:
            recommendations.append("💉 **Iniciar programa de suplementación** - El paciente no recibe actualmente suplementos")
        
        if sis == 0:
            recommendations.append("🏥 **Considerar afiliación al SIS** para seguimiento y tratamiento gratuito")
        
        recommendations.append("📅 **Control de hemoglobina** en 1-2 meses para evaluar respuesta al tratamiento")
        
    else:  # Sin anemia
        recommendations.append("✅ **Continuar con alimentación balanceada** rica en hierro y vitaminas")
        recommendations.append("📊 **Control preventivo** de hemoglobina cada 6-12 meses")
        
        if edad_meses < 36:
            recommendations.append("👶 **Niños pequeños**: Mantener lactancia materna y/o alimentación complementaria adecuada")
        
        if suplementacion == 1:
            recommendations.append("💊 **Mantener suplementación** preventiva según indicación médica")
        
        if prob_anemia > 0.4:
            recommendations.append("⚡ **Riesgo moderado detectado** - Reforzar medidas preventivas y monitorear en 3 meses")
    
    return recommendations


def calculate_zscore_interpretation(ptz, zte, zpe):
    """Interpreta los Z-scores nutricionales."""
    interpretations = []
    
    # PTZ - Peso para Talla
    if ptz < -3:
        interpretations.append("📉 **Desnutrición aguda severa** (PTZ < -3)")
    elif ptz < -2:
        interpretations.append("📉 **Desnutrición aguda moderada** (PTZ -3 a -2)")
    elif ptz > 2:
        interpretations.append("📈 **Sobrepeso** (PTZ > 2)")
    elif ptz > 3:
        interpretations.append("📈 **Obesidad** (PTZ > 3)")
    else:
        interpretations.append("✅ **Peso/Talla normal** (PTZ -2 a +2)")
    
    # ZTE - Talla para Edad
    if zte < -2:
        interpretations.append("📏 **Talla baja/Desnutrición crónica** (ZTE < -2)")
    else:
        interpretations.append("✅ **Talla normal para edad** (ZTE ≥ -2)")
    
    return interpretations


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Verificar que existen los modelos
    if not os.path.exists('models/best_model.joblib'):
        st.error("⚠️ No se encontró el modelo. Ejecuta primero `python save_model.py`")
        st.stop()
    
    # Cargar modelo
    model, scaler, model_info = load_model()
    
    # Header
    st.markdown('<p class="main-header">🔬 Sistema de Predicción de Anemia</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar con información del modelo
    with st.sidebar:
        st.header("ℹ️ Información del Modelo")
        st.info(f"""
        **Modelo:** {model_info['model_name']}  
        **Técnica:** {model_info['balancing_technique']}  
        **Muestras entrenamiento:** {model_info['training_samples']:,}
        """)
        
        st.header("📋 Variables de Entrada")
        st.markdown("""
        - **Sexo**: Masculino/Femenino
        - **Edad**: En meses (0-60)
        - **Peso**: En kilogramos
        - **Talla**: En centímetros
        - **Z-scores**: PTZ, ZTE, ZPE
        - **Altitud**: En metros
        - **Programas**: Suplementación, SIS
        """)
    
    # Formulario de entrada
    st.header("📝 Datos del Paciente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Datos Básicos")
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino"], index=0)
        edad_meses = st.slider("Edad (meses)", 0, 60, 24)
        peso = st.number_input("Peso (kg)", min_value=1.0, max_value=30.0, value=12.0, step=0.1)
        talla = st.number_input("Talla (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.5)
    
    with col2:
        st.subheader("📊 Z-Scores")
        ptz = st.slider("PTZ (Peso/Talla)", -5.0, 5.0, 0.0, 0.1, 
                        help="Z-score Peso para Talla")
        zte = st.slider("ZTE (Talla/Edad)", -5.0, 5.0, 0.0, 0.1,
                        help="Z-score Talla para Edad")
        zpe = st.slider("ZPE (Peso/Edad)", -5.0, 5.0, 0.0, 0.1,
                        help="Z-score Peso para Edad")
    
    with col3:
        st.subheader("📍 Otros Datos")
        altura = st.number_input("Altitud (msnm)", min_value=0, max_value=5000, value=3000, step=100,
                                 help="Altitud del lugar de residencia")
        suplementacion = st.checkbox("Recibe Suplementación", value=False)
        sis = st.checkbox("Tiene SIS", value=True, 
                          help="Seguro Integral de Salud")
    
    st.markdown("---")
    
    # Botón de predicción
    if st.button("🔍 Realizar Predicción", type="primary", use_container_width=True):
        # Preparar features
        sexo_encoded = 1 if sexo == "Masculino" else 0
        features = [
            sexo_encoded,
            edad_meses,
            peso,
            talla,
            ptz,
            zte,
            zpe,
            altura,
            1 if suplementacion else 0,
            1 if sis else 0
        ]
        
        # Realizar predicción
        prediction, probability = predict_anemia(model, scaler, features)
        
        # Mostrar resultado
        st.header("📊 Resultado")
        
        col_result, col_prob = st.columns([2, 1])
        
        with col_result:
            if prediction == 1:
                st.markdown("""
                <div class="result-box anemia-positive">
                    🔴 RIESGO DE ANEMIA DETECTADO
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box anemia-negative">
                    🟢 SIN RIESGO DE ANEMIA
                </div>
                """, unsafe_allow_html=True)
        
        with col_prob:
            st.metric("Probabilidad de Anemia", f"{probability[1]*100:.1f}%")
            st.metric("Confianza del Modelo", f"{max(probability)*100:.1f}%")
        
        # Barra de probabilidad
        st.progress(probability[1], text=f"Probabilidad de anemia: {probability[1]*100:.1f}%")
        
        # Interpretación Z-scores
        st.header("📏 Estado Nutricional")
        interpretations = calculate_zscore_interpretation(ptz, zte, zpe)
        for interp in interpretations:
            st.markdown(interp)
        
        # Recomendaciones
        st.header("💡 Recomendaciones")
        recommendations = get_recommendations(
            prediction, 
            probability[1], 
            edad_meses, 
            1 if suplementacion else 0,
            1 if sis else 0
        )
        
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f"• {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Disclaimer
        st.warning("""
        ⚠️ **IMPORTANTE**: Este sistema es una herramienta de apoyo y NO reemplaza el diagnóstico médico profesional. 
        El diagnóstico definitivo de anemia requiere un examen de hemoglobina en sangre realizado por un profesional de salud.
        """)


if __name__ == "__main__":
    main()

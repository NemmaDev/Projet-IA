import streamlit as st
import numpy as np
import joblib
import os

# Vérification de l'existence des fichiers
def check_files():
    files_needed = ["best_model.pkl", "scaler.pkl"]
    missing_files = [f for f in files_needed if not os.path.exists(f)]
    return missing_files

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Maladie Cardiaque",
    page_icon="🩺",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        margin-top: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Vérification des fichiers nécessaires
missing_files = check_files()
if missing_files:
    st.error(f"Erreur : Les fichiers suivants sont manquants : {', '.join(missing_files)}")
    st.info("Assurez-vous d'avoir exécuté le notebook heart_disease_analysis.ipynb pour générer ces fichiers.")
    st.stop()

# Chargement du modèle et du scaler
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Interface principale
st.title("🩺 Prédiction de Maladie Cardiaque")
st.markdown("""
<div class="info-box" style="background-color: #f0f2f6;">
Cette application utilise l'apprentissage automatique pour évaluer le risque de maladie cardiaque
en se basant sur différents paramètres médicaux.
</div>
""", unsafe_allow_html=True)

# Création de deux colonnes pour les inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Informations démographiques et vitales")
    age = st.slider("Âge", 29, 77, 50)
    sex = st.radio("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
    trestbps = st.number_input("Pression artérielle au repos (mm Hg)", 90, 200, 120)
    chol = st.number_input("Cholestérol sérique (mg/dl)", 100, 600, 200)
    fbs = st.radio("Glycémie à jeun > 120 mg/dl", [0, 1], 
                   format_func=lambda x: "Non" if x == 0 else "Oui")

with col2:
    st.subheader("Paramètres cardiaques")
    cp = st.selectbox("Type de douleur thoracique", 
                     options=[0, 1, 2, 3],
                     format_func=lambda x: {
                         0: "Angine typique",
                         1: "Angine atypique",
                         2: "Douleur non angineuse",
                         3: "Asymptomatique"
                     }[x])
    
    restecg = st.selectbox("Résultats ECG au repos", 
                          options=[0, 1, 2],
                          format_func=lambda x: {
                              0: "Normal",
                              1: "Anomalie ST-T",
                              2: "Hypertrophie ventriculaire"
                          }[x])
    
    thalach = st.slider("Fréquence cardiaque maximale", 70, 220, 150)
    exang = st.radio("Angine induite par l'exercice", [0, 1],
                     format_func=lambda x: "Non" if x == 0 else "Oui")
    oldpeak = st.slider("Dépression ST à l'exercice", 0.0, 6.2, 1.0, 0.1)
    
    slope = st.selectbox("Pente du segment ST", 
                        options=[0, 1, 2],
                        format_func=lambda x: {
                            0: "Ascendante",
                            1: "Plate",
                            2: "Descendante"
                        }[x])
    
    ca = st.selectbox("Nombre de vaisseaux majeurs colorés", [0, 1, 2, 3])
    thal = st.selectbox("Thalassémie", 
                       options=[3, 6, 7],
                       format_func=lambda x: {
                           3: "Normal",
                           6: "Défaut fixe",
                           7: "Défaut réversible"
                       }[x])

# Bouton de prédiction
if st.button("Analyser le risque cardiaque"):
    with st.spinner("Analyse en cours..."):
        # Préparation des données
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]])
        # Standardisation
        features_scaled = scaler.transform(features)
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        # Affichage des résultats
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ Risque détecté (Probabilité: {proba[1]:.1%})")
            st.markdown("""
                <div style="background-color: #ffe5e5; padding: 1rem; border-radius: 0.5rem;">
                Le modèle suggère un risque élevé de maladie cardiaque. Une consultation médicale est recommandée.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success(f"✅ Risque faible (Probabilité: {proba[0]:.1%})")
            st.markdown("""
                <div style="background-color: #e5ffe5; padding: 1rem; border-radius: 0.5rem;">
                Le modèle ne détecte pas de risque significatif de maladie cardiaque.
                </div>
                """, unsafe_allow_html=True)
        
        # Affichage des probabilités détaillées
        st.markdown("### 📊 Niveau de Risque Cardiaque")
        risk_percentage = int(proba[1] * 100)
        st.progress(risk_percentage)

        st.write("\n📊 Détail des probabilités :")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilité absence de maladie", f"{proba[0]:.1%}")
        with col2:
            st.metric("Probabilité présence de maladie", f"{proba[1]:.1%}")

# Avertissement
st.markdown("---")
st.markdown("""
<div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; font-size: 0.9em;">
⚠️ <strong>Note importante</strong> : Cette application est un outil d'aide à la décision et ne remplace en aucun cas 
l'avis d'un professionnel de santé. Consultez toujours un médecin pour un diagnostic médical.
</div>
""", unsafe_allow_html=True)

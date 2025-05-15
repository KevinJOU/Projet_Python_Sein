import streamlit as st
import pandas as pd
import numpy as np
import joblib 

st.set_page_config(layout="wide", page_title="Prédiction Tumeur", page_icon="🧬")

st.markdown( """<style>.stApp {background-image: url("https://github.com/KevinJOU/Projet_Python_Sein/blob/main/Image.png");
#         background-size: cover;}
#     </style> """, unsafe_allow_html=True)


# # Charger le modèle de regression et le scaler
modele_rf = joblib.load('modele_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

# Je nomme mes variables principales
feature_names = ['concave points_mean', 'concave points_worst', 'radius_mean', 'concavity_mean', 'concavity_worst', 'area_se']

# Titre de mon application
st.title("Prédiction de la malignité d'une tumeur du sein")

# Titre de ma section d'entrée des données
st.write("Ajustez les valeurs des variables ci-dessous afin d'obtenenir " \
"la prédiction de la malignité de la tumeur.")    
   
# Collecte des entrées utilisateur via la sidebar
user_inputs = []
concave_points_mean = st.number_input("concave points_mean", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
concave_points_worst = st.number_input("concave points_worst", min_value=0.0, max_value=0.3, value=0.0, step=0.01)
radius_mean = st.number_input("radius mean", min_value=0.0, max_value=30.0, value=0.0, step=5.0)
concavity_mean= st.number_input("concavity_mean", min_value=0.0, max_value=0.50, value=0.0, step=0.01)
concavity_worst= st.number_input("concavity_worst", min_value=0.0, max_value=1.25, value=0.0, step=0.01)
area_se = st.number_input("area_se", min_value=0.0, max_value=550.0, value=0.0, step=20.0)
#st.write("le nombre entré est", area_se)


# Création du dataframe pour la prédiction
if st.button("Prédire"):
    # Je créé mon tableau de collecte de données avec les valeurs dans le bon ordre
    input_data = np.array([[concave_points_mean, concave_points_worst, 
                            radius_mean, concavity_mean, concavity_worst, area_se]])
    
    # Standardisation des données
    scaled_data = scaler.transform(input_data)
    
    # Prédiction
    prediction = modele_rf.predict(scaled_data)
    proba = modele_rf.predict_proba(scaled_data)
    
    # Affichage des résultats
    st.subheader("Résultat de la prédiction :")
    
    if prediction[0] == 1:
        st.error(f"Tumeur maligne détectée (probabilité : {proba[0][1]*100:.2f}%)")
    else:
        st.success(f"Tumeur bénigne détectée (probabilité : {proba[0][0]*100:.2f}%)")

    # Affichage détaillé des probabilités
    st.write(f"Détail des probabilités :")
    st.write(f"- Bénigne : {proba[0][0]*100:.2f}%")
    st.write(f"- Maligne : {proba[0][1]*100:.2f}%")

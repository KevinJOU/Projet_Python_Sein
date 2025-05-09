import streamlit as st
import pandas as pd
import numpy as np
import joblib


# # Charger le modèle de regression et le scaler
modele_reg = joblib.load('modele_regression.pkl')
scaler = joblib.load('scaler.pkl')

# Je nomme mes variables principales
feature_names = [
    "radius_mean", 
    "concave points_mean",
    "radius_se",
    "area_se", 
    "texture_worst", 
    "concavity_worst",
]
# Titre de mon application
st.title("Prédiction de la malignité d'une tumeur du sein")

# Titre de ma section d'entrée des données
st.write("Ajustez les valeurs des variables ci-dessous afin d'obtenenir " \
"la prédiction de la malignité de la tumeur.")    
   
# Collecte des entrées utilisateur via la sidebar
user_inputs = []
radius_mean = st.number_input("Radius mean", min_value=0.0, max_value=30.0, value=0.0, step=0.01)
concave_points_mean = st.number_input("concave points_mean", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
radius_se = st.number_input("radius_se", min_value=0.0, max_value=5.0, value=0.0, step=0.01)
area_se = st.number_input("area_se", min_value=0.0, max_value=550.0, value=0.0, step=0.01)
texture_worst = st.number_input("Texture worst", min_value=0.0, max_value=50.0, value=0.0, step=0.01)
concavity_worst= st.number_input("concavity_worst", min_value=0.0, max_value=1.25, value=0.0, step=0.01)
#st.write("le nombre entré est", radius_mean)

# Création du dataframe pour la prédiction
if st.button("Prédire"):
    # Je créé mon tableau de collecte de données avec les valeurs dans le bon ordre
    input_data = np.array([[
        radius_mean, 
        concave_points_mean, 
        radius_se, 
        area_se, 
        texture_worst,
        concavity_worst
]])
    
    # Standardisation des données
    scaled_data = scaler.transform(input_data)
    
    # Prédiction
    prediction = modele_reg.predict(scaled_data)
    proba = modele_reg.predict_proba(scaled_data)
    
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





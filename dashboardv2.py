import pandas as pd
import numpy as np
from PIL import Image , ImageDraw , ImageFont
import cv2 
import streamlit as st
import pydeck as pdk
from io import BytesIO 
import imageio
import requests
import json
import geocoder
from PIL import Image

########################################################## APPEL API #####################################################################################

API_URL_PREDICT="https://cleancity-nanterre-ddlzhsitgq-ew.a.run.app//predict"
# Les données d'exemple que l'on utilise si on n'a pas de points


########################################################## PAGE DE GARDE ET LOCALISATION ###################################################################

samples = pd.DataFrame([
    np.asarray([48.822507, 2.268754]) + np.random.randn(2) * 0.004 for _ in range(50)
], columns=['lat', 'lon'])

# Les points où se trouvent les déchets
if "points" not in st.session_state:
    points = pd.DataFrame([], columns=['lat', 'lon'])
else:
    points = pd.DataFrame.from_dict(st.session_state["points"])
    
st.set_page_config(layout="wide", page_title="CleanCity")
st.write("## CleanCity")
tab_predict2,tab_predict, tab_map = st.tabs(["Détection par appareil photo","Détection importé", "Map"])



########################################################## CONSTRUCTION DES PAGES ###################################################################


tab_predict.write(
    "Vous pouvez charger n'importe quelle image afin de détecter les déchets sur celle-ci. Pour cela, veuillez charger une image dans le menu de gauche."
)
tab_predict2.write("Vous pouvez prendre en photo les déchets directement depuis votre caméra ")


########################################################## Mise en page de l'appareil photo ###################################################################

captured_img=None
with tab_predict2:
    # Initialize session state
    if 'capture' not in st.session_state:
        st.session_state['capture'] = False

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    ret, frame = cap.read()  # Read a frame from the camera

    if ret:
        st.image(frame, channels="BGR")

    if st.button("Capture"):
        st.session_state['capture'] = True
        captured_img = frame.copy()
        st.image(captured_img, channels="BGR")
        # You can process `captured_img` here

    cap.release()


##########################################################  Converssion de la matrice image en image PNG ###################################################################

# Specify the path for saving the PNG file
#output_path = 'C:\\Users\\Aziz\\Desktop\\CleanCity\\output.png'

# Save the NumPy array as a PNG file
#imageio.imwrite(output_path, captured_img)



########################################################## LOCALISATION ###################################################################


heatmap = tab_map.checkbox("Afficher la heatmap ?", value=True)
use_samples = tab_map.checkbox("Utiliser les données d'exemple ?", value=False)

tab_map.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=48.900359,
        longitude=2.226477,
        zoom=14
    ),
	layers=[
	    pdk.Layer(
	        'HeatmapLayer',
	        data=samples if use_samples else points,
	        get_position='[lon, lat]'
	    )
	    if heatmap else
	    pdk.Layer(
	        'ColumnLayer',
	        data=samples if use_samples else points,
	        get_position='[lon, lat]',
            radius=20,
            elevationScale=0,
            get_color="[255, 0, 0, 255]"
	    )
	]
))

########################################################## Fonction utiles au projet #####################################################################################

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def get_latitude():
    try:
        # Utilisez la fonction get() de geocoder avec l'argument 'gps' pour obtenir les coordonnées GPS
        location = geocoder.ip('me').latlng

        # Vérifiez si les coordonnées ont été récupérées avec succès
        if location:
            latitude, longitude = location
            return latitude
        else:
            print("Impossible de récupérer les coordonnées GPS.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

def get_longitude():
    try:
        # Utilisez la fonction get() de geocoder avec l'argument 'gps' pour obtenir les coordonnées GPS
        location = geocoder.ip('me').latlng

        # Vérifiez si les coordonnées ont été récupérées avec succès
        if location:
            latitude, longitude = location
            return longitude
        else:
            print("Impossible de récupérer les coordonnées GPS.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

###################################################### Fonction detect pour un drag and drop ######################################################################
def detect(upload):
    image = Image.open(upload)
    col1.write("Image d'origine :camera:")
    col1.image(image)
    
    files = {'file':  convert_image(image)}
    response = requests.post(API_URL_PREDICT, files=files)
    predictions = json.loads(response.content)
    
    # Image avec les prédictions
    draw = ImageDraw.Draw(image, "RGBA")
    img_fraction = image.size[1] / 3200
    font = ImageFont.truetype("arial.ttf", int(max(15, 60 * img_fraction)))
    for p in predictions:
        x, y, w, h, proba = p[1]
        
        # Réduire la bounding box tout en maintenant son centre
        reduction_factor = 0.8  # Ajustez cette valeur en fonction de votre besoin
        new_w = int(w * reduction_factor)
        new_h = int(h * reduction_factor)
        
        # Calculer les nouvelles coordonnées du coin supérieur gauche
        new_x = x + (w - new_w) // 2
        new_y = y + (h - new_h) // 2
        
        # Dessiner le rectangle sur l'image
        draw.rectangle([new_x, new_y, new_x + new_w, new_y + new_h], outline="red", width=3)
    
    col2.write("Image avec les déchets détectés :wrench:")
    col2.image(image)
    return predictions

###################################################### Fonction detect pour une photo ######################################################################

def detect2(upload):
    image = Image.open(upload)
    #colonne1.write("Image d'origine :camera:")
    #colonne1.image(image)
    
    files = {'file':  convert_image(image)}
    response = requests.post(API_URL_PREDICT, files=files)
    predictions = json.loads(response.content)
    
    # Image avec les prédictions
    draw = ImageDraw.Draw(image, "RGBA")
    img_fraction = image.size[1] / 3200
    font = ImageFont.truetype("arial.ttf", int(max(15, 60 * img_fraction)))
    for p in predictions:
        x, y, w, h, proba = p[1]
    
        
        # Dessiner le rectangle sur l'image
        draw.rectangle([x, y, x + w/2, y + h/2], outline="red", width=3)
    
    colonne1.write("Image avec les déchets détectés :wrench:")
    colonne1.image(image)
    return predictions




########################################################## PAGE DE GARDE ET LOCALISATION ###################################################################


#file_upload = tab_predict.file_uploader("Charger une image", type=["png", "jpg", "jpeg"])
col1, col2 = tab_predict.columns(2)

colonne1, colonne2 = tab_predict.columns(2)

########################################################## Effectuer la prédiction du upload  ################################################

#if file_upload is not None:

#    predictions = detect(upload=file_upload)
#    tab_predict.write("#### Résultats")
#    for p in predictions:
#        tab_predict.write("- **{}** (probabilité {:2.1f}%)".format(p[0], p[1][-1] * 100))
#    tab_predict.write("#### Quelles sont les coordonnées de la photo ?")
#    cols = tab_predict.columns(2)
#    latitude = cols[0].number_input("Latitude", value=48.822507, min_value=0.0, max_value=90.0)
#    longitude = cols[1].number_input("Longitude", value=2.268754, min_value=-180.0, max_value=180.0)
#    validate_coords = tab_predict.button("Valider les coordonnées")

########################################################## Effectuer la prédiction de output ####################################################

if captured_img is not None:
# Specify the path for saving the PNG file
    output_path = 'C:\\Users\\mvyas\\OneDrive - Hiram Finance\\Bureau\\clean_city\\output.png'
    # Save the NumPy array as a PNG file
    print(type(output_path))

    imageio.imwrite(output_path, captured_img)

    # Open the JPEG file using PIL
    #image_pil = Image.open(output_path)

    predictions = detect2(upload=output_path)
    tab_predict.write("#### Résultats")
    for p in predictions:
        tab_predict.write("- **{}** (probabilité {:2.1f}%)".format(p[0], p[1][-1] * 100))
    #tab_predict2.write("#### Quelles sont les coordonnées de la photo ?")
    #cols = tab_predict2.columns(2)
    #latitude = cols[0].number_input("Latitude", value=48.822507, min_value=0.0, max_value=90.0)
    #longitude = cols[1].number_input("Longitude", value=2.268754, min_value=-180.0, max_value=180.0)
    #validate_coords = tab_predict2.button("Valider les coordonnées")




#if validate_coords:
#    tab_predict.success("Image ajoutée dans la Map !")
#    points = pd.concat((
#        points,
#        pd.DataFrame([[latitude, longitude]], columns=['lat', 'lon'])
#    ), ignore_index=True)
 #   st.session_state["points"] = points.to_dict()


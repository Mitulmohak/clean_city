import pandas as pd
import numpy as np
from PIL import Image
import cv2 
import streamlit as st
import pydeck as pdk

API_URL_PREDICT="https://cdupropre-mitul-ddlzhsitgq-od.a.run.app///predict"

# Les données d'exemple que l'on utilise si on n'a pas de points
samples = pd.DataFrame([
    np.asarray([48.822507, 2.268754]) + np.random.randn(2) * 0.004 for _ in range(50)
], columns=['lat', 'lon'])

# Les points où se trouvent les déchets
if "points" not in st.session_state:
    points = pd.DataFrame([], columns=['lat', 'lon'])
else:
    points = pd.DataFrame.from_dict(st.session_state["points"])
    
st.set_page_config(layout="wide", page_title="CleanCity" )
#st.set_page_config(layout="wide", page_title="Détection de déchets", background_color="#00c2b0")

st.write("# CleanCity")
#menu=st.button("Menu", type="primary")
# Object notation
# "with" notation


# Variable de suivi pour l'état du menu
menu_open = False

# Bouton pour ouvrir/fermer le menu
if st.button("Menu", type="primary"):
    # Inverser l'état du menu à chaque clic
    menu_open = not menu_open

# Afficher le menu si l'état est True
if menu_open:
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    
    with col1:
        st.button("Classement")
    with col2:
        st.button("Mon score")
    with col3:
        st.button("Guide de tri des déchets")
    with col4:
        st.button("Mes infos")
    with col5:
        st.button("En savoir plus sur nous")



tab_predict, tab_map = st.tabs(["Détection","Map"])
tab_predict.write(
    "Vous pouvez charger n'importe quelle image afin de détecter les déchets sur celle-ci. Pour cela, veuillez charger une image dans le menu de gauche."
)
cap = cv2.VideoCapture(0)

    # Configuration de la taille de la vidéo
cap.set(3, 640)  # Largeur
cap.set(4, 480)  # Hauteur
while True:
    ret, frame = cap.read()  # Lire une image depuis la caméra

    # Afficher l'image dans le tableau de bord
    st.image(frame, channels="BGR")

    # Option pour arrêter la capture
    stop_capture = st.button("Arrêter la capture")

    if stop_capture:
        break

# Libérer la capture vidéo à la fin
cap.release()

    # Boucle principale du tableau de bord

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


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


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
    	# TODO : Dessiner des rectangles sur l'images
        draw.rectangle([x,y,x+w,y+h],outline="red",width =3)
    col2.write("Image avec les déchets détectés :wrench:")
    col2.image(image)
    return predictions

file_upload = tab_predict.file_uploader("Charger une image", type=["png", "jpg", "jpeg"])
col1, col2 = tab_predict.columns(2)

if file_upload is not None:
    predictions = detect(upload=file_upload)
    tab_predict.write("#### Résultats")
    for p in predictions:
        tab_predict.write("- **{}** (probabilité {:2.1f}%)".format(p[0], p[1][-1] * 100))
    tab_predict.write("#### Quelles sont les coordonnées de la photo ?")
    cols = tab_predict.columns(2)
    latitude = cols[0].number_input("Latitude", value=48.822507, min_value=0.0, max_value=90.0)
    longitude = cols[1].number_input("Longitude", value=2.268754, min_value=-180.0, max_value=180.0)
    validate_coords = tab_predict.button("Valider les coordonnées")

    if validate_coords:
        tab_predict.success("Image ajoutée dans la Map !")
        points = pd.concat((
            points,
            pd.DataFrame([[latitude, longitude]], columns=['lat', 'lon'])
        ), ignore_index=True)
        st.session_state["points"] = points.to_dict()


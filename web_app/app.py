from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import easyocr

# On va import ton code d'analyse depuis main.py
# (idéalement, tu factorises la partie "analyse_image" dans une fonction)
# Pour l'exemple, je vais directement reprendre la partie "analyse"

# Si main.py est dans le dossier parent, on peut faire :
# from main import analyze_image  (à créer !)
# Mais pour ce code minimal, on va juste coller la partie d'analyse.

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Initialiser l'OCR (attention, si tu utilises GPU=True, assure-toi d'avoir CUDA)
reader = easyocr.Reader(['en'], gpu=False)

@app.route('/')
def index():
    return render_template('index.html')  # simple page HTML avec un formulaire

@app.route('/upload', methods=['POST'])
def upload():
    # Récupérer le fichier depuis le formulaire
    file = request.files.get('file')
    if not file:
        return "No file uploaded.", 400

    # Sauvegarde temporaire
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Appeler la fonction d'analyse
    result_data = analyze_cs2_image(filepath)

    # On pourrait afficher le résultat dans une page HTML
    # Soit on redirige vers un template, soit on renvoie direct du HTML
    return render_template('result.html', **result_data)

def analyze_cs2_image(img_path):
    """
    ICI : on réutilise la logique de main.py pour :
      - Charger l'image
      - Détecter la map
      - Extraire timer, score, etc.
    Retourne un dict qu'on pourra injecter dans result.html
    """
    # => Simplifié : Charger l'image, appeller la pipeline "gagnante" (combo_4_none_r2.0) etc.
    # => Je mets ici un pseudo-code très minimal :
    
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "Impossible de lire l'image."}

    # ... Code pour classer la map (Ancient/Nuke/Anubis) ...
    detected_map = "Nuke"   # ex, ou on appelle ta fonction classify_map(...)
    timer = "1:23"
    ct_score = 4
    t_score = 8

    # ICI TU REPRENDS LA PARTIE "combo_4_none_r2.0"
    # - zoom, ROI, half top, resize x2, OCR, etc. 
    # - tu renvoies timer, ct_score, t_score, detected_map

    return {
      "detected_map": detected_map,
      "timer": timer,
      "ct_score": ct_score,
      "t_score": t_score
    }

if __name__ == "__main__":
    app.run(debug=True)

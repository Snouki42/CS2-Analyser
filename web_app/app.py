import os
import cv2
import numpy as np
import easyocr
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for

###############################
# Paramètres GLOBAUX
###############################
BINS = 32
ROI_TOP = 0.00
ROI_BOTTOM = 0.20
ROI_LEFT = 0.40
ROI_RIGHT = 0.60
CROP_SIDE_LEFT = 90
CROP_SIDE_RIGHT = 90
KEEP_TOP_HALF = True

# Config Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Init EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

###############################
# Fonctions Utilitaires
###############################
def ensure_upload_dir():
    """Crée le dossier uploads si nécessaire."""
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print(f"[DEBUG] Upload directory: {app.config['UPLOAD_FOLDER']}")

def save_image_in_uploads(filename, img):
    """Enregistre une image dans le répertoire uploads."""
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(path, img)
    print(f"[DEBUG] Saved => {path}")
    return path

def list_files(folder):
    exts = (".jpg", ".png", ".jpeg")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

###############################
# Fonctions pour Signature de Map
###############################
def compute_hist(image, bins=BINS):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

def build_map_signature(image_paths, bins=BINS):
    signature = np.zeros((bins, bins), dtype=np.float32)
    count = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            signature += compute_hist(img, bins)
            count += 1
    return signature / count if count > 0 else signature

def classify_map(img, map_signatures, bins=BINS):
    hist_img = compute_hist(img, bins)
    best_map, best_score = None, -999
    for map_name, signature in map_signatures.items():
        score = cv2.compareHist(hist_img, signature, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_map, best_score = map_name, score
    return best_map, best_score

###############################
# Chargement des signatures de map
###############################
def init_map_signatures():
    map_paths = {
        "Ancient": list_files("dataset/ancient"),
        "Nuke": list_files("dataset/nuke"),
        "Anubis": list_files("dataset/anubis"),
        "Dust2": list_files("dataset/dust2"),
        "Inferno": list_files("dataset/inferno"),
        "Mirage": list_files("dataset/mirage"),
        "Train": list_files("dataset/train")
    }
    return {name: build_map_signature(paths) for name, paths in map_paths.items()}

###############################
# Analyse de l'image CS2
###############################
def analyze_cs2_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return {"error": f"Impossible de lire {img_path}"}

    # 1) Classification de la map
    detected_map, _ = classify_map(img, map_signatures)

    # 2) Extraction du scoreboard
    h, w = img.shape[:2]
    top, bottom, left, right = int(ROI_TOP * h), int(ROI_BOTTOM * h), int(ROI_LEFT * w), int(ROI_RIGHT * w)
    scoreboard_roi = img[top:bottom, left:right]

    scoreboard_narrow = scoreboard_roi[:, CROP_SIDE_LEFT:scoreboard_roi.shape[1] - CROP_SIDE_RIGHT]
    if KEEP_TOP_HALF:
        scoreboard_narrow = scoreboard_narrow[:scoreboard_narrow.shape[0] // 2]

    # 3) Prétraitement et OCR
    gray = cv2.cvtColor(scoreboard_narrow, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(gray, detail=1)

    texts = [r[1] for r in results]
    timer, ct_score, t_score = "??:??", -1, -1
    if len(texts) >= 3:
        timer = texts[0].replace('.', ':') if '.' in texts[0] else texts[0]
        try:
            ct_score, t_score = int(texts[1]), int(texts[2])
        except ValueError:
            pass

    # Générer l'image annotée
    dbg_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (coords, txt, _) in results:
        x_min, y_min = int(min(p[0] for p in coords)), int(min(p[1] for p in coords))
        x_max, y_max = int(max(p[0] for p in coords)), int(max(p[1] for p in coords))
        cv2.rectangle(dbg_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(dbg_img, txt, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    debug_filename = os.path.basename(img_path).replace('.', '_annot.')
    save_image_in_uploads(debug_filename, dbg_img)

    return {
        "detected_map": detected_map,
        "timer": timer,
        "ct_score": ct_score,
        "t_score": t_score,
        "original_image_url": f"/uploads/{os.path.basename(img_path)}",
        "debug_image_url": f"/uploads/{debug_filename}"
    }

###############################
# Routes Flask
###############################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    ensure_upload_dir()  # S'assurer que le dossier existe avant la sauvegarde
    
    try:
        file.save(filepath)
        print(f"[DEBUG] Fichier sauvegardé : {filepath}")
    except Exception as e:
        print(f"[ERROR] Erreur lors de la sauvegarde : {e}")
        return jsonify({"error": f"File save failed: {str(e)}"}), 500

    result_data = analyze_cs2_image(filepath)
    return jsonify(result_data)


@app.route('/uploads/<filename>')
def serve_uploads(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"[ERROR] Failed to serve file {filename}: {e}")
        return f"File not found: {filename}", 404
###############################
# Lancement Flask
###############################
if __name__ == "__main__":
    ensure_upload_dir()
    map_signatures = init_map_signatures()
    app.run(debug=True)

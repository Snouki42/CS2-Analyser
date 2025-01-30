import os
import cv2
import numpy as np
import easyocr

from flask import Flask, request, render_template, jsonify

###############################
# Paramètres GLOBAUX du pipeline
###############################
DEBUG = True
DEBUG_DIR = "debug_cs2_project"

# 1) Pour la détection de MAP (Ancient, Nuke, Anubis)
BINS = 32

# 2) Pour la zone scoreboard
ROI_TOP    = 0.00
ROI_BOTTOM = 0.20
ROI_LEFT   = 0.40
ROI_RIGHT  = 0.60

CROP_SIDE_LEFT  = 90
CROP_SIDE_RIGHT = 90
KEEP_TOP_HALF   = True

# Liste combos (pour info)
RESIZE_FACTORS = [1.0, 2.0]
THRESHOLDS = [
    ("none",       False, False),
    ("binary",     True,  False),
    ("binary_inv", True,  True)
]
THRESH_VAL = 160

###############################
# Flask
###############################
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Init EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # si tu as CUDA; sinon gpu=False

###############################
# Fonctions pipeline
###############################
def ensure_debug_dir():
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

def debug_save(filename, img):
    path = os.path.join(DEBUG_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[DEBUG] Saved => {path}")

def compute_hist(image, bins=32):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[bins,bins],[0,180,0,256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def classify_map(img, map_signatures, bins=32):
    hist_img = compute_hist(img, bins)
    best_map = None
    best_score = -999
    for map_name, sig in map_signatures.items():
        score = cv2.compareHist(hist_img, sig, cv2.HISTCMP_CORREL)
        if score>best_score:
            best_score = score
            best_map = map_name
    return best_map, best_score

def build_map_signature(image_paths, bins=32):
    signature = np.zeros((bins,bins), dtype=np.float32)
    count=0
    for p in image_paths:
        im = cv2.imread(p)
        if im is None:
            print(f"[WARN] Impossible de lire {p}")
            continue
        h = compute_hist(im, bins)
        signature += h
        count+=1
    if count>0:
        signature /= count
    return signature

def list_files(folder):
    exts = (".jpg",".png",".jpeg")
    allp = []
    if not os.path.isdir(folder):
        print(f"[WARN] Dossier introuvable : {folder}")
        return allp
    for fname in os.listdir(folder):
        fpath = os.path.join(folder,fname)
        if os.path.isfile(fpath) and fpath.lower().endswith(exts):
            allp.append(fpath)
    return allp

###############################
# Charger la signature de map au démarrage
###############################
map_signatures = {}
def init_map_signatures():
    # ex. dataset/ancient, dataset/nuke, dataset/anubis
    ancient_paths = list_files("../dataset/ancient")
    nuke_paths    = list_files("../dataset/nuke")
    anubis_paths  = list_files("../dataset/anubis")

    sig_ancient = build_map_signature(ancient_paths, bins=BINS)
    sig_nuke    = build_map_signature(nuke_paths,    bins=BINS)
    sig_anubis  = build_map_signature(anubis_paths,  bins=BINS)

    return {
        "Ancient": sig_ancient,
        "Nuke":    sig_nuke,
        "Anubis":  sig_anubis
    }

###############################
# Pipeline final
###############################
def analyze_cs2_image(img_path):
    """
    Exécute la logique 'combo_4_none_r2.0'
    + classification de map
    + extraction du timer, score
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"error": f"Impossible de lire {img_path}"}

    # 1) Map
    detected_map, map_score = classify_map(img, map_signatures, bins=BINS)

    # 2) ROI scoreboard
    h, w = img.shape[:2]
    top    = int(ROI_TOP*h)
    bottom = int(ROI_BOTTOM*h)
    left   = int(ROI_LEFT*w)
    right  = int(ROI_RIGHT*w)
    scoreboard_roi = img[top:bottom, left:right].copy()

    # Retrait 90 px
    sb_h, sb_w = scoreboard_roi.shape[:2]
    new_left  = CROP_SIDE_LEFT
    new_right = sb_w - CROP_SIDE_RIGHT
    if new_right <= new_left:
        return {"error": "recadrage trop large => new_right<=new_left."}

    scoreboard_narrow = scoreboard_roi[:, new_left:new_right].copy()

    # Conserver la moitié haute
    if KEEP_TOP_HALF:
        nh, nw = scoreboard_narrow.shape[:2]
        scoreboard_narrow = scoreboard_narrow[:nh//2, :]

    # On applique "combo_4_none_r2.0" => gris, resize x2, pas de threshold
    gray = cv2.cvtColor(scoreboard_narrow, cv2.COLOR_BGR2GRAY)
    resize_factor=2.0
    new_w = int(gray.shape[1]*resize_factor)
    new_h = int(gray.shape[0]*resize_factor)
    gray = cv2.resize(gray, (new_w,new_h), interpolation=cv2.INTER_CUBIC)

    # OCR
    results = reader.readtext(gray, detail=1)

    texts_list = [r[1] for r in results]
    # ex. on attend au moins 3 blocs => timer, ct_score, t_score
    if len(texts_list) < 3:
        return {
          "detected_map": detected_map,
          "timer": "??:??",
          "ct_score": -1,
          "t_score": -1
        }

    timer = texts_list[0]
    # Correction . => :
    if '.' in timer and len(timer)<=4:
        timer=timer.replace('.',':')

    ct_str = texts_list[1]
    t_str  = texts_list[2]
    try:
        ct_score = int(ct_str)
        t_score  = int(t_str)
    except:
        ct_score=-1
        t_score=-1

    return {
      "detected_map": detected_map,
      "timer": timer,
      "ct_score": ct_score,
      "t_score": t_score
    }

###############################
# ROUTES FLASK
###############################
@app.route('/')
def index():
    # Page HTML -> templates/index.html
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Récupérer le fichier
    file = request.files.get('file')
    if not file:
        return jsonify({"error":"No file"}), 400

    # Sauvegarde
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Analyse
    result_data = analyze_cs2_image(filepath)
    return jsonify(result_data)

###############################
# LANCEMENT
###############################
if __name__=="__main__":
    # 1) Créer le dossier uploads/
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # 2) Créer le dossier debug
    ensure_debug_dir()
    # 3) Charger la signature des maps
    map_signatures = init_map_signatures()

    # 4) Lancer Flask
    app.run(debug=True)

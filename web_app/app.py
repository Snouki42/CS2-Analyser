import os
import cv2
import numpy as np
import easyocr
import re
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for

###############################
# PARAMÈTRES GLOBAUX
###############################
DEBUG = True
DEBUG_DIR = "C:\WorkSpace\ISEN\CS\CS2-Analyser\web_app\static\debug_cs2_project"
CONFIDENCE_THRESHOLD = 0.12

# 1) Pour la détection de MAP (Ancient, Nuke, Anubis)
BINS = 32
# On suppose un dossier dataset/ancient, dataset/nuke, dataset/anubis
# Chacun contenant quelques images représentatives.

# 2) Pour la zone scoreboard
ROI_TOP = 0.00
ROI_BOTTOM = 0.25
ROI_LEFT = 0.20
ROI_RIGHT = 0.80

# Retirer 90 px à gauche/droite
CROP_SIDE_LEFT = 90
CROP_SIDE_RIGHT = 90

# Conserver la moitié haute
KEEP_TOP_HALF = True

# 3) Listes de combos
RESIZE_FACTORS = [1.0, 2.0]
THRESHOLDS = [
    ("none", False, False),
    ("binary", True, False),
    ("binary_inv", True, True)
]
THRESH_VAL = 160

# Config Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Init EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

###############################
# FONCTIONS DE DEBUG
###############################
def ensure_debug_dir():
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)

def debug_save(filename, img):
    """Sauvegarde l'image dans le dossier debug_cs2_project/"""
    path = os.path.join(DEBUG_DIR, filename)
    cv2.imwrite(path, img)
    print(f"[DEBUG] Saved => {path}")

###############################
# DÉTECTION DE MAP (HISTOGRAMMES)
###############################
def compute_hist(image, bins=BINS):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def classify_map(img, map_signatures, bins=BINS):
    hist_img = compute_hist(img, bins)
    best_map = None
    best_score = -999
    for map_name, sig in map_signatures.items():
        score = cv2.compareHist(hist_img, sig, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best_map = map_name
    return best_map, best_score

def build_map_signature(image_paths, bins=BINS):
    signature = np.zeros((bins, bins), dtype=np.float32)
    count = 0
    for p in image_paths:
        im = cv2.imread(p)
        if im is None:
            print(f"[WARN] Impossible de lire {p}")
            continue
        h = compute_hist(im, bins)
        signature += h
        count += 1
    if count > 0:
        signature /= count
    return signature

def init_map_signatures():
    map_paths = {
        "Ancient": list_files("dataset/ancient"),
        "Nuke": list_files("dataset/nuke"),
        "Anubis": list_files("dataset/anubis"),
        "Dust2": list_files("dataset/dust2"),
        "Mirage": list_files("dataset/mirage"),
        "Train": list_files("dataset/train")
    }
    return {name: build_map_signature(paths) for name, paths in map_paths.items()}

###############################
# OCR
# Pensez à dl CUDA pour carte Nvidia pour profiter de la puissance du GPU
###############################
reader = easyocr.Reader(['en'], gpu=True)

###############################
# MAIN
###############################
def analyze_cs2_image(img_path):
    timer_passed = False

    ensure_debug_dir()

    # 1) Construction des signatures pour Ancient, Nuke, Anubis
    # dataset/ancient/*.jpg, dataset/nuke/*.jpg, dataset/anubis/*.jpg
    ancient_paths = list_files("dataset/ancient")
    nuke_paths = list_files("dataset/nuke")
    anubis_paths = list_files("dataset/anubis")
    dust2_paths = list_files("dataset/dust2")
    mirage_paths = list_files("dataset/mirage")
    train_paths = list_files("dataset/train")

    sig_ancient = build_map_signature(ancient_paths, bins=BINS)
    sig_nuke = build_map_signature(nuke_paths, bins=BINS)
    sig_anubis = build_map_signature(anubis_paths, bins=BINS)
    sig_dust2 = build_map_signature(dust2_paths, bins=BINS)
    sig_mirage = build_map_signature(mirage_paths, bins=BINS)
    sig_train = build_map_signature(train_paths, bins=BINS)

    map_signatures = {
        "Ancient": sig_ancient,
        "Nuke": sig_nuke,
        "Anubis": sig_anubis,
        "Dust2": sig_dust2,
        "Mirage": sig_mirage,
        "Train": sig_train,
    }

    # 2) Lecture de l'image
    img = cv2.imread(img_path)
    if img is None:
        return {"error": f"Impossible de lire {img_path}"}
    debug_save("0_input_image.png", img)

    # 3) Classify la map
    detected_map, map_score = classify_map(img, map_signatures, bins=BINS)
    print(f"[MAP] Détectée = {detected_map} (score corr={map_score:.3f})")

    # 4) Rognage scoreboard (ROI)
    h, w = img.shape[:2]
    top = int(ROI_TOP * h)
    bottom = int(ROI_BOTTOM * h)
    left = int(ROI_LEFT * w)
    right = int(ROI_RIGHT * w)

    scoreboard_roi = img[top:bottom, left:right].copy()
    debug_save("1_scoreboard_roi.png", scoreboard_roi)

    # Retirer 90 px
    sb_h, sb_w = scoreboard_roi.shape[:2]
    new_left = CROP_SIDE_LEFT
    new_right = sb_w - CROP_SIDE_RIGHT
    if new_right <= new_left:
        return {"error": "recadrage trop large => new_right<=new_left."}
    scoreboard_narrow = scoreboard_roi[:, new_left:new_right].copy()
    debug_save("2_scoreboard_narrow.png", scoreboard_narrow)

    # Conserver la moitié haute
    if KEEP_TOP_HALF:
        nh, nw = scoreboard_narrow.shape[:2]
        scoreboard_narrow = scoreboard_narrow[:nh // 2, :]
    debug_save("3_scoreboard_top_half.png", scoreboard_narrow)

    # 5) Multi-prétraitement combos
    combos_found = []
    combo_index = 0

    # Configuration pour combo_4_none_r2.0
    resize_factor = 2.0
    use_thresh = False
    invert = False
    combo_label = "combo_4_none_r2.0"

    # a) Conversion gris
    gray = cv2.cvtColor(scoreboard_narrow, cv2.COLOR_BGR2GRAY)

    # b) Resize
    if resize_factor != 1.0:
        new_w = int(gray.shape[1] * resize_factor)
        new_h = int(gray.shape[0] * resize_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # c) Threshold
    proc = gray

    proc_name = f"{combo_label}.png"
    debug_save(proc_name, proc)

    # d) OCR
    results = reader.readtext(proc, detail=1)
    # Annot
    dbg = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR) if len(proc.shape) == 2 else proc.copy()
    filtered_results = []

    for (coords, txt, conf) in results:
        print(f"txt: {txt}")
        if conf < CONFIDENCE_THRESHOLD:
            continue  # Skip results below the confidence threshold
        if '.' in txt or '-' in txt or ':' in txt or timer_passed:
            timer_passed = True
            print(f"timer passed: {timer_passed}")
        else:
            continue

        if timer_passed:
            filtered_results.append((coords, txt, conf))

    for (coords, txt, conf) in filtered_results:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        print(f"text : {txt} ")
        txt = txt.replace('.', ':')
        txt = txt.replace('-', ':')
        txt = txt.replace('S', '')
        print(f"text changé : {txt} ")

        cv2.rectangle(dbg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(dbg, txt, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"[INFO FINI] {combo_label} => {len(filtered_results)} blocs =>", [r[1] for r in filtered_results])
    dbg_name = f"{combo_label}_annot.png"
    debug_save(dbg_name, dbg)

    # e) Log les textes
    texts_list = [r[1].replace('.', ':').replace('-', ':').replace('S', '').replace('s', '') for r in filtered_results]
    print(f"[INFO] {combo_label} => {len(filtered_results)} blocs =>", texts_list)
    timer = texts_list[0]
    ct_score = int(texts_list[1])
    t_score = int(texts_list[2])
    if len(texts_list) >= 13:
        if not texts_list[3].endswith('0'):
            texts_list.pop(3)
            if not texts_list[3].endswith('0'):
                texts_list.pop(3)

        for i in range(3, 13):
            texts_list[i] = re.sub(r'\D', '0', texts_list[i])

            value = int(texts_list[i])
            if value > 16000:
                texts_list[i] = texts_list[i][1:]  # Remove the first character

        ct_economie = sum(int(text) for text in texts_list[3:8])
        t_economie = sum(int(text) for text in texts_list[8:13])
    else:
        ct_economie = 0
        t_economie = 0

    print(f"tableau final : {texts_list}")

    print(f"CT Économie: {ct_economie}, T Économie: {t_economie}")

    print(f"Timer: {timer}, CT Score: {ct_score}, T Score: {t_score}")

    # Fin
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"Timer: {timer}, CT Score: {ct_score}, T Score: {t_score}, sur la map {detected_map}")

    return {
        "detected_map": detected_map,
        "timer": timer,
        "ct_score": ct_score,
        "t_score": t_score,
        "texts_list": texts_list,
        "ct_economie": ct_economie,
        "t_economie": t_economie
    }

###############################
# HELPER pour liste_fichiers
###############################
def list_files(folder):
    """Retourne la liste de .jpg/.png dans le dossier."""
    exts = (".jpg", ".png", ".jpeg")
    allp = []
    if not os.path.isdir(folder):
        print(f"[WARN] Dossier introuvable : {folder}")
        return allp
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath) and fpath.lower().endswith(exts):
            allp.append(fpath)
    return allp

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
    ensure_debug_dir()  # S'assurer que le dossier existe avant la sauvegarde

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
    ensure_debug_dir()
    map_signatures = init_map_signatures()
    app.run(debug=True)
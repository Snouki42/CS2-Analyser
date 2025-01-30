import cv2
import numpy as np
import os
import easyocr

###############################
# PARAMÈTRES GLOBAUX
###############################

DEBUG = True
DEBUG_DIR = "debug_cs2_project"

# 1) Pour la détection de MAP (Ancient, Nuke, Anubis)
BINS = 32
MAP_NAMES = ["Ancient", "Nuke", "Anubis"]
# On suppose un dossier dataset/ancient, dataset/nuke, dataset/anubis
# Chacun contenant quelques images représentatives.

# 2) Pour la zone scoreboard
ROI_TOP    = 0.00
ROI_BOTTOM = 0.20
ROI_LEFT   = 0.40
ROI_RIGHT  = 0.60

# Retirer 90 px à gauche/droite
CROP_SIDE_LEFT  = 90
CROP_SIDE_RIGHT = 90

# Conserver la moitié haute
KEEP_TOP_HALF   = True

# 3) Listes de combos
RESIZE_FACTORS = [1.0, 2.0]
THRESHOLDS = [
    ("none",       False, False),
    ("binary",     True,  False),
    ("binary_inv", True,  True)
]
THRESH_VAL = 160

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
    hist = cv2.calcHist([hsv],[0,1],None,[bins,bins],[0,180,0,256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def classify_map(img, map_signatures, bins=BINS):
    hist_img = compute_hist(img, bins)
    best_map = None
    best_score = -999
    for map_name, sig in map_signatures.items():
        score = cv2.compareHist(hist_img, sig, cv2.HISTCMP_CORREL)
        if score>best_score:
            best_score = score
            best_map = map_name
    return best_map, best_score

def build_map_signature(image_paths, bins=BINS):
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

###############################
# OCR
# Pensez à dl CUDA pour carte Nvidia pour profiter de la puissance du GPU
###############################
reader = easyocr.Reader(['en'], gpu=True)

###############################
# MAIN
###############################
def main():
                            
    ensure_debug_dir()

    # 1) Construction des signatures pour Ancient, Nuke, Anubis
    # dataset/ancient/*.jpg, dataset/nuke/*.jpg, dataset/anubis/*.jpg
    ancient_paths = list_files("dataset/ancient")
    nuke_paths    = list_files("dataset/nuke")
    anubis_paths  = list_files("dataset/anubis")
    dust2_paths = list_files("dataset/dust2")
    inferno_paths = list_files("dataset/inferno")
    mirage_paths = list_files("dataset/mirage")
    train_paths = list_files("dataset/train")

    sig_ancient = build_map_signature(ancient_paths, bins=BINS)
    sig_nuke    = build_map_signature(nuke_paths,    bins=BINS)
    sig_anubis  = build_map_signature(anubis_paths,  bins=BINS)
    sig_dust2  = build_map_signature(dust2_paths,  bins=BINS)
    sig_inferno  = build_map_signature(inferno_paths,  bins=BINS)
    sig_mirage = build_map_signature(mirage_paths,  bins=BINS)
    sig_train  = build_map_signature(train_paths,  bins=BINS)

    map_signatures = {
        "Ancient": sig_ancient,
        "Nuke":    sig_nuke,
        "Anubis":  sig_anubis,
        "Dust2":  sig_dust2,
        "Inferno":  sig_inferno,
        "Mirage":  sig_mirage,
        "Train":  sig_train,
    }

    # 2) Lecture de l'image
    img_path = "1.png"  # A adapter
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERREUR] Impossible de lire {img_path}")
        return
    debug_save("0_input_image.png", img)

    # 3) Classify la map
    detected_map, map_score = classify_map(img, map_signatures, bins=BINS)
    print(f"[MAP] Détectée = {detected_map} (score corr={map_score:.3f})")

    # 4) Rognage scoreboard (ROI)
    h, w = img.shape[:2]
    top    = int(ROI_TOP*h)
    bottom = int(ROI_BOTTOM*h)
    left   = int(ROI_LEFT*w)
    right  = int(ROI_RIGHT*w)

    scoreboard_roi = img[top:bottom, left:right].copy()
    debug_save("1_scoreboard_roi.png", scoreboard_roi)

    # Retirer 90 px
    sb_h, sb_w = scoreboard_roi.shape[:2]
    new_left  = CROP_SIDE_LEFT
    new_right = sb_w - CROP_SIDE_RIGHT
    if new_right <= new_left:
        print("[ERREUR] recadrage trop large => new_right<=new_left.")
        return
    scoreboard_narrow = scoreboard_roi[:, new_left:new_right].copy()
    debug_save("2_scoreboard_narrow.png", scoreboard_narrow)

    # Conserver la moitié haute
    if KEEP_TOP_HALF:
        nh, nw = scoreboard_narrow.shape[:2]
        scoreboard_narrow = scoreboard_narrow[:nh//2, :]
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
    for (coords, txt, conf) in results:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        if '.' in txt and len(txt) <= 4:
            txt = txt.replace('.', ':')

        cv2.rectangle(dbg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(dbg, txt, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    dbg_name = f"{combo_label}_annot.png"
    debug_save(dbg_name, dbg)

    # e) Log les textes
    texts_list = [r[1] for r in results]
    print(f"[INFO] {combo_label} => {len(results)} blocs =>", texts_list)

    if len(texts_list) >= 3:
        timer = texts_list[0].replace('.', ':')
        ct_score = int(texts_list[1])
        t_score = int(texts_list[2])
    
    print(f"Timer: {timer}, CT Score: {ct_score}, T Score: {t_score}")

    # Fin
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"Timer: {timer}, CT Score: {ct_score}, T Score: {t_score}, sur la map {detected_map}")


###############################
# HELPER pour liste_fichiers
###############################
def list_files(folder):
    """Retourne la liste de .jpg/.png dans le dossier."""
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
# LANCEMENT
###############################
if __name__=="__main__":
    main()
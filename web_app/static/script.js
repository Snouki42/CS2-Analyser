const dropzone = document.getElementById("dropzone");
const resultDiv = document.getElementById("result");

// Afficher un message de chargement
function showLoading() {
    resultDiv.innerHTML = `
        <div class="loading">
            <p>Chargement en cours...</p>
        </div>
    `;
}

// Afficher une erreur
function showError(message) {
    console.error("Erreur:", message);
    resultDiv.innerHTML = `
        <div class="error" style="color: red; padding: 10px;">
            <p>Erreur: ${message}</p>
        </div>
    `;
}

// Afficher les résultats
function showResults(data) {
    console.log("Résultats reçus:", data);
    const html = `
        <h2>Résultat</h2>
        <p>Map détectée : ${data.detected_map}</p>
        <p>Timer : ${data.timer}</p>
        <p>Score CT-T : ${data.ct_score} - ${data.t_score}</p>
        <p>Argent CT-T : ${data.ct_economie}$ - ${data.t_economie}$</p>
        <div class="images">
            <div class="image-container">
                <h3>Image originale</h3>
                <img src="/static/debug_cs2_project/0_input_image.png" 
                     alt="Original" 
                     style="max-width: 100%;"
                     onerror="this.onerror=null; showError('Impossible de charger l\'image originale')"/>
            </div>
            <div class="image-container">
                <h3>Image annotée</h3>
                 <img src="/static/debug_cs2_project/combo_4_none_r2.0_annot.png" 
                     alt="Debug" 
                     style="max-width: 100%;"
                     onerror="this.onerror=null; showError('Impossible de charger l\'image annotée')"/>
            </div>
        </div>
    `;
    resultDiv.innerHTML = html;
}

// Gestionnaire de drag & drop
function initializeDragAndDrop() {
    const events = ["dragenter", "dragover", "dragleave", "drop"];
    
    events.forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    // Effet visuel pendant le drag
    ["dragenter", "dragover"].forEach(event => {
        dropzone.addEventListener(event, () => {
            dropzone.classList.add("highlight");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        dropzone.addEventListener(event, () => {
            dropzone.classList.remove("highlight");
        });
    });

    // Gestion du drop
    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFile(files[0]);
        }
    });

    // Gestion du click pour sélection de fichier
    dropzone.addEventListener("click", () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = (e) => {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        };
        input.click();
    });
}

// Fonction d'upload
async function uploadFile(file) {
    if (!file.type.startsWith('image/')) {
        showError("Veuillez sélectionner une image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    
    showLoading();
    
    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            showResults(data);
        }
    } catch (error) {
        showError(`Erreur lors de l'upload: ${error.message}`);
    }
}

// Initialisation
document.addEventListener("DOMContentLoaded", () => {
    initializeDragAndDrop();
    // Message initial
    dropzone.innerHTML = `
        <div class="dropzone-content">
            <p>Glissez une image ici ou cliquez pour sélectionner</p>
            <p class="small">Formats acceptés: .jpg, .png, .jpeg</p>
        </div>
    `;
});
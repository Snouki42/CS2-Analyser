const dropzone = document.getElementById("dropzone");
const resultDiv = document.getElementById("result");

// Empêche le comportement par défaut (ouvrir l'image dans le browser) sur dragover & drop
["dragenter","dragover","dragleave","drop"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
  });
});

// Ajouter un style highlight quand on survole la zone
["dragenter","dragover"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    dropzone.classList.add("highlight");
  });
});
["dragleave","drop"].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    dropzone.classList.remove("highlight");
  });
});

// Gérer le drop
dropzone.addEventListener("drop", e => {
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    const file = files[0];
    uploadFile(file);
  }
});

function uploadFile(file) {
  // On crée un FormData
  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      // Erreur
      resultDiv.innerHTML = `<p style="color:red;">Erreur: ${data.error}</p>`;
    } else {
      // On affiche le résultat
      const html = `
        <h2>Résultat</h2>
        <p>Map détectée : ${data.detected_map}</p>
        <p>Timer : ${data.timer}</p>
        <p>Score CT-T : ${data.ct_score} - ${data.t_score}</p>
      `;
      resultDiv.innerHTML = html;
    }
  })
  .catch(err => {
    console.error("Erreur drag & drop", err);
    resultDiv.innerHTML = `<p style="color:red;">Erreur durant l'upload!</p>`;
  });
}

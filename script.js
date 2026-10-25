const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const imagePreview = document.getElementById("imagePreview");
const fileInfo = document.getElementById("fileInfo");
const errorMessage = document.getElementById("errorMessage");

const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultContent = document.getElementById("resultContent");

const predictedCondition = document.getElementById("predictedCondition");
const confidenceScore = document.getElementById("confidenceScore");
const confidenceBar = document.getElementById("confidenceBar");
const probabilityList = document.getElementById("probabilityList");

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const allowedTypes = [
  "image/jpeg",
  "image/png",
  "image/webp"
];

// Fixed example values for the frontend demonstration only.
const demoResults = [
  { name: "Coronary Artery Disease", score: 4 },
  { name: "Cardiomyopathy", score: 3 },
  { name: "Heart Failure", score: 2 },
  { name: "Myocardial Infarction", score: 2 },
  { name: "Arrhythmia", score: 3 },
  { name: "Valvular Heart Disease", score: 2 },
  { name: "Congenital Heart Disease", score: 2 },
  { name: "Normal / No Abnormality", score: 82 }
];

let selectedFile = null;
let previewUrl = null;
let isAnalyzing = false;
let analysisTimer = null;

function showError(message) {
  errorMessage.textContent = message;
}

function clearError() {
  showError("");
}

function clearPreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  imagePreview.removeAttribute("src");
  imagePreview.hidden = true;
  uploadPlaceholder.hidden = false;
}

function showEmptyState() {
  emptyState.hidden = false;
  loadingState.hidden = true;
  resultContent.hidden = true;
}

function resetResults() {
  if (analysisTimer !== null) {
    clearTimeout(analysisTimer);
    analysisTimer = null;
  }

  isAnalyzing = false;
  analyzeBtn.disabled = !selectedFile;
  analyzeBtn.innerHTML = "<span>✦</span> Analyze Image";

  showEmptyState();
}

function handleFile(file) {
  if (isAnalyzing) {
    return;
  }

  clearError();

  if (!file) {
    return;
  }

  if (!allowedTypes.includes(file.type)) {
    imageInput.value = "";
    selectedFile = null;
    fileInfo.textContent = "No image selected";
    clearPreview();
    resetResults();
    showError("Please select a JPG, JPEG, PNG, or WEBP image.");
    analyzeBtn.disabled = true;
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    imageInput.value = "";
    selectedFile = null;
    fileInfo.textContent = "No image selected";
    clearPreview();
    resetResults();
    showError("The image must be smaller than 10 MB.");
    analyzeBtn.disabled = true;
    return;
  }

  const newUrl = URL.createObjectURL(file);

  const testImage = new Image();

  testImage.onload = function () {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    previewUrl = newUrl;
    selectedFile = file;

    imagePreview.src = previewUrl;
    imagePreview.hidden = false;
    uploadPlaceholder.hidden = true;

    fileInfo.textContent =
      file.name + " · " + (file.size / 1024).toFixed(1) + " KB";

    analyzeBtn.disabled = false;

    resetResults();
  };

  testImage.onerror = function () {
    URL.revokeObjectURL(newUrl);
    imageInput.value = "";
    selectedFile = null;
    fileInfo.textContent = "No image selected";
    clearPreview();
    resetResults();
    analyzeBtn.disabled = true;
    showError("This image could not be opened. Please choose another image.");
  };

  testImage.src = newUrl;
}

function renderResults(data = null) {
  const results = data && Array.isArray(data.probabilities) && data.probabilities.length
    ? data.probabilities
    : demoResults;

  const bestResult = results.reduce((best, item) => {
    return item.score > best.score ? item : best;
  }, results[0] || { name: "Normal / No Abnormality", score: 0 });

  const predictedName = data && data.predicted_condition ? data.predicted_condition : bestResult.name;
  const confidenceValue = data && typeof data.confidence !== "undefined" ? data.confidence : bestResult.score;

  predictedCondition.textContent = predictedName;
  confidenceScore.textContent = confidenceValue + "%";
  confidenceBar.style.width = confidenceValue + "%";

  probabilityList.replaceChildren();

  results.forEach(function (item) {
    const row = document.createElement("div");
    row.className = "probability-item";

    const meta = document.createElement("div");
    meta.className = "probability-meta";

    const name = document.createElement("span");
    name.className = "probability-name";
    name.textContent = item.name;

    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = item.score + "%";

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = item.score + "%";

    meta.append(name, value);
    track.appendChild(fill);
    row.append(meta, track);
    probabilityList.appendChild(row);
  });

  emptyState.hidden = true;
  loadingState.hidden = true;
  resultContent.hidden = false;
}

imageInput.addEventListener("change", function () {
  const file = imageInput.files[0];
  handleFile(file);
});

analyzeBtn.addEventListener("click", function () {
  if (!selectedFile || isAnalyzing) {
    return;
  }

  clearError();
  isAnalyzing = true;
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
  imageInput.disabled = true;
  analyzeBtn.textContent = "Processing...";

  emptyState.hidden = true;
  resultContent.hidden = true;
  loadingState.hidden = false;

  const formData = new FormData();
  formData.append("image", selectedFile);

  fetch("/analyze", {
    method: "POST",
    body: formData
  })
    .then(function (response) {
      return response.json().then(function (data) {
        return {
          ok: response.ok,
          data: data
        };
      });
    })
    .then(function (result) {
      if (!result.ok) {
        throw new Error(result.data.error || "Analysis failed.");
      }

      renderResults(result.data);
      isAnalyzing = false;
      analysisTimer = null;

      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = "<span>✦</span> Analyze Again";
      resetBtn.disabled = false;
      imageInput.disabled = false;
    })
    .catch(function (error) {
      console.error(error);
      showError(error.message || "The analysis request could not be completed.");
      showEmptyState();

      isAnalyzing = false;
      analysisTimer = null;

      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = "<span>✦</span> Analyze Image";
      resetBtn.disabled = false;
      imageInput.disabled = false;
    });
});

resetBtn.addEventListener("click", function () {
  imageInput.value = "";
  selectedFile = null;

  clearPreview();
  clearError();
  resetResults();

  fileInfo.textContent = "No image selected";
  analyzeBtn.disabled = true;
  resetBtn.disabled = false;
  imageInput.disabled = false;
  uploadArea.scrollIntoView({
    behavior: "smooth",
    block: "center"
  });
});
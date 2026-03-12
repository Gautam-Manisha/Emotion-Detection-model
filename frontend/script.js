/**
 * script.js – EmotiVision Detect Page
 * Handles:
 *  - File upload (click & drag-drop)
 *  - Webcam start / capture / stop
 *  - POST /predict API call
 *  - Result rendering (emoji, confidence bar, breakdown grid)
 *  - Loading spinner & error handling
 */

'use strict';

// ─── Config ──────────────────────────────────────────────────────────────────
const API_URL = 'http://localhost:5000/predict';

// Emotion → emoji mapping
const EMOTION_EMOJI = {
  Angry:    '😠',
  Disgust:  '🤢',
  Fear:     '😨',
  Happy:    '😊',
  Sad:      '😢',
  Surprise: '😲',
  Neutral:  '😐',
};

// Emotion → accent colour for breakdown bars
const EMOTION_COLOUR = {
  Angry:    '#f43f5e',
  Disgust:  '#84cc16',
  Fear:     '#a855f7',
  Happy:    '#f59e0b',
  Sad:      '#60a5fa',
  Surprise: '#fb923c',
  Neutral:  '#94a3b8',
};

// ─── State ────────────────────────────────────────────────────────────────────
let selectedFile = null;
let webcamStream = null;

// ─── Tab Switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.id === `tab-${name}`);
    t.setAttribute('aria-selected', t.id === `tab-${name}`);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === `panel-${name}`);
  });

  // Clean up the other panel
  if (name === 'upload') stopWebcam();
  else clearUpload();

  hideResult();
  hideError();
}

// ─── Upload Handlers ──────────────────────────────────────────────────────────
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) setFile(file);
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
}

function handleDragOver(event) {
  event.preventDefault();
  document.getElementById('upload-zone').classList.add('drag-over');
}

function handleDragLeave() {
  document.getElementById('upload-zone').classList.remove('drag-over');
}

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById('preview-img');
    previewImg.src = e.target.result;
    document.getElementById('preview-wrap').hidden = false;
    document.getElementById('upload-zone').style.display = 'none';
  };
  reader.readAsDataURL(file);

  document.getElementById('analyze-btn').disabled = false;
  hideError();
  hideResult();
}

function clearUpload() {
  selectedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('preview-img').src = '';
  document.getElementById('preview-wrap').hidden = true;
  document.getElementById('upload-zone').style.display = '';
  document.getElementById('analyze-btn').disabled = true;
  hideResult();
  hideError();
}

// ─── Webcam Handlers ──────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const video = document.getElementById('webcam-video');
    video.srcObject = webcamStream;
    video.hidden = false;

    document.getElementById('webcam-overlay').classList.add('hidden');
    document.getElementById('start-cam-btn').disabled = true;
    document.getElementById('capture-btn').disabled = false;
    document.getElementById('stop-cam-btn').hidden = false;

    hideResult();
    hideError();
  } catch (err) {
    showError('Camera access denied or not available. Please allow camera permission and try again.');
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  const video = document.getElementById('webcam-video');
  video.srcObject = null;
  video.hidden = true;

  document.getElementById('webcam-overlay').classList.remove('hidden');
  document.getElementById('start-cam-btn').disabled = false;
  document.getElementById('capture-btn').disabled = true;
  document.getElementById('stop-cam-btn').hidden = true;
  document.getElementById('capture-preview-wrap').hidden = true;
}

async function captureAndAnalyze() {
  const video   = document.getElementById('webcam-video');
  const canvas  = document.getElementById('webcam-canvas');
  const context = canvas.getContext('2d');

  // Draw video frame onto canvas
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Show captured frame preview
  const captureImg = document.getElementById('capture-img');
  captureImg.src = canvas.toDataURL('image/jpeg');
  document.getElementById('capture-preview-wrap').hidden = false;

  // Convert canvas to a Blob (JPEG)
  canvas.toBlob(async (blob) => {
    const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
    await sendToAPI(file);
  }, 'image/jpeg', 0.92);
}

// ─── Analysis ─────────────────────────────────────────────────────────────────
async function analyzeUpload() {
  if (!selectedFile) return;
  await sendToAPI(selectedFile);
}

async function sendToAPI(file) {
  showSpinner();
  hideError();
  hideResult();

  try {
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `Server error: ${response.status}`);
    }

    renderResult(data);
  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.name === 'TypeError') {
      showError('Cannot reach the backend. Make sure the Flask server is running on http://localhost:5000');
    } else {
      showError(err.message);
    }
  } finally {
    hideSpinner();
  }
}

// ─── Result Rendering ─────────────────────────────────────────────────────────
function renderResult(data) {
  const { emotion, confidence, all_emotions, bbox_image } = data;

  // Main emotion label + emoji
  document.getElementById('result-emoji').textContent   = EMOTION_EMOJI[emotion] ?? '🤔';
  document.getElementById('result-emotion').textContent = emotion;

  // Confidence bar
  const pct = Math.round(confidence * 100);
  document.getElementById('confidence-value').textContent = `${pct}%`;
  const bar = document.getElementById('confidence-bar');
  bar.style.width = '0%';
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { bar.style.width = `${pct}%`; });
  });

  // Annotated image
  const resultImg = document.getElementById('result-img');
  if (bbox_image) {
    resultImg.src = `http://localhost:5000${bbox_image}`;
    resultImg.alt = `Face with detected emotion: ${emotion}`;
  } else {
    document.querySelector('.result-img-wrap').hidden = true;
  }

  // Breakdown grid
  const grid = document.getElementById('breakdown-grid');
  grid.innerHTML = '';
  Object.entries(all_emotions)
    .sort(([, a], [, b]) => b - a)
    .forEach(([label, score], idx) => {
      const scorePct = Math.round(score * 100);
      const colour   = EMOTION_COLOUR[label] ?? '#6366f1';
      const isTop    = label === emotion;

      const item = document.createElement('div');
      item.className = 'breakdown-item';
      item.style.animationDelay = `${idx * 0.06}s`;
      if (isTop) item.style.borderColor = colour + '55';
      item.innerHTML = `
        <div class="breakdown-top">
          <span>${EMOTION_EMOJI[label] ?? ''} ${label}</span>
          <span class="breakdown-score">${scorePct}%</span>
        </div>
        <div class="breakdown-bar-wrap">
          <div class="breakdown-bar" style="background:${colour}; width:0%;" data-target="${scorePct}"></div>
        </div>`;
      grid.appendChild(item);
    });

  // Animate breakdown bars after a tick
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.breakdown-bar').forEach(b => {
        b.style.width = `${b.dataset.target}%`;
      });
    });
  });

  document.getElementById('result-panel').hidden = false;
  document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetAll() {
  hideResult();
  clearUpload();
  stopWebcam();
  switchTab('upload');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── UI Helpers ───────────────────────────────────────────────────────────────
function showSpinner()  { document.getElementById('spinner').hidden = false; }
function hideSpinner()  { document.getElementById('spinner').hidden = true;  }
function hideResult()   { document.getElementById('result-panel').hidden = true; }
function hideError()    { document.getElementById('error-box').hidden = true; }
function closeError()   { hideError(); }

function showError(message) {
  document.getElementById('error-msg').textContent = message;
  document.getElementById('error-box').hidden = false;
  document.getElementById('error-box').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

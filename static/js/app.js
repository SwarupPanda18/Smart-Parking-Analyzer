/**
 * Mother's Hut Traffic & Gate Operations Console
 * Client Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // 0. Auto-detecting & Configurable Backend API Endpoint
  function getApiBase() {
    const customUrl = localStorage.getItem("backend_api_url");
    if (customUrl) return customUrl.replace(/\/+$/, "");
    // Connect to port 8000 when opened via file:// or dev servers (e.g. 5500, 3000)
    if (window.location.protocol === "file:" || (window.location.port !== "8000" && window.location.port !== "")) {
      return "http://127.0.0.1:8000";
    }
    return "";
  }
  const API_BASE = getApiBase();

  // Application State
  const state = {
    selectedFile: null,
    selectedSampleUrl: null,
    activeTab: "upload", // 'upload' | 'samples' | 'webcam'
    lastResult: null,
    detectedPlates: [],
    zoomLevel: 1.0,
    webcamStream: null,
    cameraFacingMode: "environment", // Default to mobile rear / back camera
    isProcessing: false,
    cctvMode: "dual", // 'dual' | 'ai' | 'raw' | 'pip'
    isLiveStreaming: false,
    liveStreamTimer: null,
    rawMediaUrl: null,
  };

  // DOM Elements
  const elements = {
    // Clock & Health
    clockDisplay: document.getElementById("clockDisplay"),
    healthStatusDot: document.getElementById("healthStatusDot"),
    healthStatusText: document.getElementById("healthStatusText"),
    deviceBadge: document.getElementById("deviceBadge"),
    themeToggleBtn: document.getElementById("themeToggleBtn"),

    // Ingest Tabs
    tabUploadBtn: document.getElementById("tabUploadBtn"),
    tabSamplesBtn: document.getElementById("tabSamplesBtn"),
    tabWebcamBtn: document.getElementById("tabWebcamBtn"),

    // Tab Containers
    uploadContainer: document.getElementById("uploadContainer"),
    samplesContainer: document.getElementById("samplesContainer"),
    webcamContainer: document.getElementById("webcamContainer"),

    // File Upload / Dropzone
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("fileInput"),
    stagedFileCard: document.getElementById("stagedFileCard"),
    stagedFileName: document.getElementById("stagedFileName"),
    stagedFileSize: document.getElementById("stagedFileSize"),
    btnRemoveFile: document.getElementById("btnRemoveFile"),

    // Samples List
    samplesList: document.getElementById("samplesList"),

    // Webcam & Mobile Back Camera
    webcamVideo: document.getElementById("webcamVideo"),
    btnToggleWebcam: document.getElementById("btnToggleWebcam"),
    btnCaptureWebcam: document.getElementById("btnCaptureWebcam"),
    cameraLensBadge: document.getElementById("cameraLensBadge"),
    btnSwitchCamera: document.getElementById("btnSwitchCamera"),
    mobileCameraInput: document.getElementById("mobileCameraInput"),

    // Controls
    confSlider: document.getElementById("confSlider"),
    confValue: document.getElementById("confValue"),
    btnProcess: document.getElementById("btnProcess"),

    // Directives & Alerts
    alertKitchen: document.getElementById("alertKitchen"),
    alertKitchenText: document.getElementById("alertKitchenText"),
    alertFloor: document.getElementById("alertFloor"),
    alertFloorText: document.getElementById("alertFloorText"),
    capacityFill: document.getElementById("capacityFill"),
    capacityText: document.getElementById("capacityText"),

    // Telemetry Stat Cards
    statTotalVehicles: document.getElementById("statTotalVehicles"),
    statLightVehicles: document.getElementById("statLightVehicles"),
    statHeavyBuses: document.getElementById("statHeavyBuses"),
    statPlatesCount: document.getElementById("statPlatesCount"),

    // Viewport Elements
    viewportEmpty: document.getElementById("viewportEmpty"),
    viewportImg: document.getElementById("viewportImg"),
    viewportVideo: document.getElementById("viewportVideo"),
    scannerOverlay: document.getElementById("scannerOverlay"),
    scannerCircleSpinner: document.getElementById("scannerCircleSpinner"),
    scannerInnerIcon: document.getElementById("scannerInnerIcon"),
    scannerTitle: document.getElementById("scannerTitle"),
    scannerStatus: document.getElementById("scannerStatus"),
    scannerElapsed: document.getElementById("scannerElapsed"),
    scannerTimerDot: document.getElementById("scannerTimerDot"),
    scannerServerBadge: document.getElementById("scannerServerBadge"),
    scannerReassurance: document.getElementById("scannerReassurance"),
    scannerErrorBox: document.getElementById("scannerErrorBox"),
    scannerErrorMessage: document.getElementById("scannerErrorMessage"),
    btnDismissScannerError: document.getElementById("btnDismissScannerError"),
    viewportMetaInfo: document.getElementById("viewportMetaInfo"),
    btnZoomIn: document.getElementById("btnZoomIn"),
    btnZoomOut: document.getElementById("btnZoomOut"),
    btnZoomReset: document.getElementById("btnZoomReset"),
    btnDownloadMedia: document.getElementById("btnDownloadMedia"),
    btnFullscreen: document.getElementById("btnFullscreen"),
    viewportContainer: document.getElementById("viewportContainer"),

    // Dual CCTV Stream Monitor Elements
    cctvRawPane: document.getElementById("cctvRawPane"),
    cctvRawVideo: document.getElementById("cctvRawVideo"),
    cctvRawImg: document.getElementById("cctvRawImg"),
    cctvRawEmpty: document.getElementById("cctvRawEmpty"),
    cctvRawMeta: document.getElementById("cctvRawMeta"),
    cctvRawFpsTag: document.getElementById("cctvRawFpsTag"),
    cctvAiPane: document.getElementById("cctvAiPane"),
    cctvAiMeta: document.getElementById("cctvAiMeta"),
    cctvLiveCanvas: document.getElementById("cctvLiveCanvas"),
    btnModeDual: document.getElementById("btnModeDual"),
    btnModeAi: document.getElementById("btnModeAi"),
    btnModeRaw: document.getElementById("btnModeRaw"),
    btnModePip: document.getElementById("btnModePip"),
    btnLiveStreamToggle: document.getElementById("btnLiveStreamToggle"),

    // Plate Ledger
    plateSearchInput: document.getElementById("plateSearchInput"),
    ledgerDateSelect: document.getElementById("ledgerDateSelect"),
    ledgerList: document.getElementById("ledgerList"),
    ledgerCountBadge: document.getElementById("ledgerCountBadge"),
    btnExportCsv: document.getElementById("btnExportCsv"),
    btnCopyAllPlates: document.getElementById("btnCopyAllPlates"),

    // System Log
    auditLogTime: document.getElementById("auditLogTime"),
    auditLogText: document.getElementById("auditLogText"),
  };

  /* ==========================================================
     1. INITIALIZATION & LIVE HEALTH TELEMETRY
     ========================================================== */
  function initClock() {
    const updateTime = () => {
      const now = new Date();
      elements.clockDisplay.textContent = now.toLocaleTimeString("en-GB", {
        hour12: false,
        timeZoneName: "short",
      });
    };
    updateTime();
    setInterval(updateTime, 1000);
  }

  async function checkBackendHealth() {
    const targetUrl = `${API_BASE}/api/health`;
    try {
      elements.healthStatusText.textContent = "CONNECTING...";
      const res = await fetch(targetUrl, { method: "GET" });
      if (res.ok) {
        const data = await res.json();
        elements.healthStatusDot.style.backgroundColor = "var(--status-ok)";
        elements.healthStatusDot.style.boxShadow = "0 0 8px var(--status-ok)";
        elements.healthStatusText.textContent = "CONNECTED";
        if (data.device) {
          elements.deviceBadge.textContent = data.device.toUpperCase();
        }
        const olInfo = (data.models && data.models.ollama_vision && data.models.ollama_vision.installed)
          ? `+ Ollama (${data.models.ollama_vision.model})`
          : "";
        logAudit(`Backend connected: 100% Local / Free ${olInfo} on ${data.device || "CPU"}.`);
      } else {
        throw new Error("HTTP " + res.status);
      }
    } catch (err) {
      elements.healthStatusDot.style.backgroundColor = "var(--status-danger)";
      elements.healthStatusDot.style.boxShadow = "0 0 8px var(--status-danger)";
      elements.healthStatusText.textContent = "DISCONNECTED";
      logAudit(`Backend offline at ${API_BASE || window.location.origin}. Start with: python app.py`);
    }
  }

  // Click health chip to recheck
  if (elements.healthStatusDot && elements.healthStatusDot.parentElement) {
    elements.healthStatusDot.parentElement.style.cursor = "pointer";
    elements.healthStatusDot.parentElement.title = "Click to re-verify backend connection";
    elements.healthStatusDot.parentElement.addEventListener("click", () => {
      checkBackendHealth();
    });
  }

  function logAudit(message) {
    const timeStr = new Date().toLocaleTimeString("en-GB", { hour12: false });
    if (elements.auditLogTime && elements.auditLogText) {
      elements.auditLogTime.textContent = `[${timeStr}]`;
      elements.auditLogText.textContent = message;
    }
  }

  /* ==========================================================
     2. THEME TOGGLE & CCTV MULTI-STREAM CONSOLE
     ========================================================== */
  function initTheme() {
    const savedTheme = localStorage.getItem("app_theme") || "dark";
    document.documentElement.setAttribute("data-theme", savedTheme);
    elements.themeToggleBtn.innerHTML = savedTheme === "dark" ? "☀️ Light" : "🌙 Dark";

    elements.themeToggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("app_theme", next);
      elements.themeToggleBtn.innerHTML = next === "dark" ? "☀️ Light" : "🌙 Dark";
    });
  }

  function initCctvConsole() {
    const savedLayout = localStorage.getItem("cctv_layout_mode") || "dual";
    setCctvLayout(savedLayout);

    if (elements.btnModeDual) elements.btnModeDual.addEventListener("click", () => setCctvLayout("dual"));
    if (elements.btnModeAi) elements.btnModeAi.addEventListener("click", () => setCctvLayout("ai"));
    if (elements.btnModeRaw) elements.btnModeRaw.addEventListener("click", () => setCctvLayout("raw"));
    if (elements.btnModePip) elements.btnModePip.addEventListener("click", () => setCctvLayout("pip"));

    if (elements.btnLiveStreamToggle) {
      elements.btnLiveStreamToggle.addEventListener("click", toggleLiveAiStream);
    }
  }

  function setCctvLayout(mode) {
    state.cctvMode = mode;
    if (elements.viewportContainer) {
      elements.viewportContainer.classList.remove("mode-dual", "mode-ai", "mode-raw", "mode-pip");
      elements.viewportContainer.classList.add(`mode-${mode}`);
    }

    if (elements.btnModeDual) elements.btnModeDual.classList.toggle("active", mode === "dual");
    if (elements.btnModeAi) elements.btnModeAi.classList.toggle("active", mode === "ai");
    if (elements.btnModeRaw) elements.btnModeRaw.classList.toggle("active", mode === "raw");
    if (elements.btnModePip) elements.btnModePip.classList.toggle("active", mode === "pip");

    localStorage.setItem("cctv_layout_mode", mode);
    logAudit(`Surveillance console layout switched to: ${mode.toUpperCase()} VIEW`);
  }

  function previewRawMedia(source, isVideo, label) {
    if (elements.cctvRawEmpty) elements.cctvRawEmpty.style.display = "none";
    if (elements.cctvRawMeta) elements.cctvRawMeta.textContent = label || "RAW OPTICAL STAGED";

    if (isVideo) {
      if (elements.cctvRawImg) elements.cctvRawImg.style.display = "none";
      if (elements.cctvRawVideo) {
        elements.cctvRawVideo.style.display = "block";
        if (typeof source === "string") {
          elements.cctvRawVideo.src = source;
        } else {
          if (state.rawMediaUrl) URL.revokeObjectURL(state.rawMediaUrl);
          state.rawMediaUrl = URL.createObjectURL(source);
          elements.cctvRawVideo.src = state.rawMediaUrl;
        }
        elements.cctvRawVideo.currentTime = 0;
      }
    } else {
      if (elements.cctvRawVideo) elements.cctvRawVideo.style.display = "none";
      if (elements.cctvRawImg) {
        elements.cctvRawImg.style.display = "block";
        if (typeof source === "string") {
          elements.cctvRawImg.src = source;
        } else {
          if (state.rawMediaUrl) URL.revokeObjectURL(state.rawMediaUrl);
          state.rawMediaUrl = URL.createObjectURL(source);
          elements.cctvRawImg.src = state.rawMediaUrl;
        }
      }
    }
  }

  function toggleLiveAiStream() {
    if (!state.webcamStream) {
      alert("Please connect the camera first before toggling Live AI Continuous Stream.");
      return;
    }
    state.isLiveStreaming = !state.isLiveStreaming;
    if (elements.btnLiveStreamToggle) {
      elements.btnLiveStreamToggle.textContent = state.isLiveStreaming ? "⚡ Live AI: Streaming (ON)" : "⚡ Live AI: Off";
      elements.btnLiveStreamToggle.style.color = state.isLiveStreaming ? "#10b981" : "";
      elements.btnLiveStreamToggle.style.borderColor = state.isLiveStreaming ? "#10b981" : "";
    }
    if (elements.cctvAiMeta) {
      elements.cctvAiMeta.textContent = state.isLiveStreaming ? "● LIVE INFERENCE STREAMING" : "0.1s MOTION GATE ACTIVE";
    }

    if (state.isLiveStreaming) {
      logAudit("Live AI stream started on CCTV camera feed.");
      runLiveStreamLoop();
    } else {
      if (state.liveStreamTimer) clearTimeout(state.liveStreamTimer);
      state.liveStreamTimer = null;
      logAudit("Live AI stream stopped.");
    }
  }

  async function runLiveStreamLoop() {
    if (!state.isLiveStreaming || !state.webcamStream || !elements.webcamVideo || !elements.webcamVideo.videoWidth) {
      state.liveStreamTimer = setTimeout(runLiveStreamLoop, 1000);
      return;
    }

    try {
      const snapCanvas = document.createElement("canvas");
      snapCanvas.width = elements.webcamVideo.videoWidth;
      snapCanvas.height = elements.webcamVideo.videoHeight;
      const ctx = snapCanvas.getContext("2d");
      ctx.drawImage(elements.webcamVideo, 0, 0, snapCanvas.width, snapCanvas.height);

      const blob = await new Promise((res) => snapCanvas.toBlob(res, "image/jpeg", 0.75));
      const conf = elements.confSlider ? parseFloat(elements.confSlider.value) : 0.40;
      const formData = new FormData();
      formData.append("file", blob, "live_stream_frame.jpg");

      const response = await fetch(`${API_BASE}/analyze-parking?conf=${conf}`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        if (result && result.status === "success" && state.isLiveStreaming) {
          renderInferenceResults(result, "LIVE_CAM_FEED", 35);
        }
      }
    } catch (e) {
      console.warn("Live stream frame error:", e);
    }

    if (state.isLiveStreaming) {
      state.liveStreamTimer = setTimeout(runLiveStreamLoop, 1500);
    }
  }

  /* ==========================================================
     3. TAB SWITCHING (UPLOAD / SAMPLES / WEBCAM)
     ========================================================== */
  function switchTab(tab) {
    state.activeTab = tab;
    elements.tabUploadBtn.classList.toggle("active", tab === "upload");
    elements.tabSamplesBtn.classList.toggle("active", tab === "samples");
    elements.tabWebcamBtn.classList.toggle("active", tab === "webcam");

    elements.uploadContainer.style.display = tab === "upload" ? "block" : "none";
    elements.samplesContainer.classList.toggle("active", tab === "samples");
    elements.webcamContainer.classList.toggle("active", tab === "webcam");

    // Stop webcam if leaving webcam tab
    if (tab !== "webcam" && state.webcamStream) {
      stopWebcam();
    }

    validateProcessButton();
  }

  elements.tabUploadBtn.addEventListener("click", () => switchTab("upload"));
  elements.tabSamplesBtn.addEventListener("click", () => {
    switchTab("samples");
    loadSamplesList();
  });
  elements.tabWebcamBtn.addEventListener("click", () => switchTab("webcam"));

  /* ==========================================================
     4. FILE INGEST & DRAG-AND-DROP
     ========================================================== */
  elements.dropzone.addEventListener("click", () => elements.fileInput.click());

  elements.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.dropzone.classList.add("dragover");
  });

  elements.dropzone.addEventListener("dragleave", () => {
    elements.dropzone.classList.remove("dragover");
  });

  elements.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  elements.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    state.selectedFile = file;
    state.selectedSampleUrl = null;
    elements.stagedFileName.textContent = file.name;
    elements.stagedFileSize.textContent = formatBytes(file.size);
    elements.stagedFileCard.classList.add("show");

    const isVideo = file.type.startsWith("video") || /\.(mp4|avi|mov|mkv)$/i.test(file.name);
    previewRawMedia(file, isVideo, `INGEST: ${file.name}`);

    validateProcessButton();
    logAudit(`File staged for analysis: ${file.name} (${formatBytes(file.size)})`);
  }

  elements.btnRemoveFile.addEventListener("click", (e) => {
    e.stopPropagation();
    state.selectedFile = null;
    elements.fileInput.value = "";
    elements.stagedFileCard.classList.remove("show");

    if (elements.cctvRawVideo) {
      elements.cctvRawVideo.src = "";
      elements.cctvRawVideo.style.display = "none";
    }
    if (elements.cctvRawImg) {
      elements.cctvRawImg.src = "";
      elements.cctvRawImg.style.display = "none";
    }
    if (elements.cctvRawEmpty) elements.cctvRawEmpty.style.display = "flex";
    if (elements.cctvRawMeta) elements.cctvRawMeta.textContent = "FEED STANDBY";

    validateProcessButton();
    logAudit("Staged file cleared.");
  });

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  /* ==========================================================
     5. SAMPLE IMAGES PICKER
     ========================================================== */
  async function loadSamplesList() {
    if (elements.samplesList.children.length > 0) return; // already loaded

    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (!res.ok) throw new Error("Could not load samples list");
      const samples = await res.json();

      elements.samplesList.innerHTML = "";
      samples.forEach((sample, index) => {
        const sampleUrl = sample.url.startsWith("http") ? sample.url : `${API_BASE}${sample.url}`;
        const item = document.createElement("div");
        item.className = "sample-item";
        item.innerHTML = `
          <img src="${sampleUrl}" alt="${sample.title}" class="sample-thumb">
          <div class="sample-info">
            <div class="sample-label">${sample.title}</div>
            <div class="sample-desc">${sample.description || "Stationary Gate Ingest"}</div>
          </div>
        `;
        item.addEventListener("click", () => {
          document.querySelectorAll(".sample-item").forEach((el) => el.classList.remove("selected"));
          item.classList.add("selected");
          state.selectedSampleUrl = sampleUrl;
          state.selectedFile = null;
          previewRawMedia(sampleUrl, false, `SAMPLE: ${sample.title}`);
          validateProcessButton();
          logAudit(`Sample feed selected: ${sample.title}`);
        });
        elements.samplesList.appendChild(item);
      });
    } catch (e) {
      elements.samplesList.innerHTML = `<div style="padding: 0.5rem; font-size: 0.75rem; color: var(--text-dim);">Unable to fetch samples from ${API_BASE}/api/samples. Start backend with: python app.py</div>`;
    }
  }

  /* ==========================================================
     6. WEBCAM STREAM & FRAME CAPTURE
     ========================================================== */
  elements.btnToggleWebcam.addEventListener("click", async () => {
    if (state.webcamStream) {
      stopWebcam();
    } else {
      await startWebcam();
    }
  });

  async function startWebcam() {
    const mode = state.cameraFacingMode || "environment";
    try {
      // Prioritize rear / back camera for mobile devices
      const constraints = {
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      state.webcamStream = stream;
      elements.webcamVideo.srcObject = stream;

      // Project directly to Main CCTV Console Raw Viewport
      if (elements.cctvRawVideo) {
        elements.cctvRawVideo.srcObject = stream;
        elements.cctvRawVideo.style.display = "block";
        if (elements.cctvRawImg) elements.cctvRawImg.style.display = "none";
        if (elements.cctvRawEmpty) elements.cctvRawEmpty.style.display = "none";
        if (elements.cctvRawMeta) elements.cctvRawMeta.textContent = "● LIVE CAM-01 STREAMING";
        if (elements.cctvRawFpsTag) elements.cctvRawFpsTag.textContent = "30 FPS · OPTICAL LIVE";
      }

      elements.btnToggleWebcam.textContent = "Disconnect Camera";
      elements.btnCaptureWebcam.disabled = false;
      if (elements.cameraLensBadge) {
        elements.cameraLensBadge.textContent = mode === "environment" 
          ? "📷 BACK CAMERA (ENVIRONMENT)" 
          : "🤳 FRONT CAMERA (USER)";
      }
      logAudit(`Live camera started with ${mode === "environment" ? "back" : "front"} lens.`);
    } catch (err) {
      // Graceful fallback for devices that don't support facingMode constraint
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        state.webcamStream = stream;
        elements.webcamVideo.srcObject = stream;

        if (elements.cctvRawVideo) {
          elements.cctvRawVideo.srcObject = stream;
          elements.cctvRawVideo.style.display = "block";
          if (elements.cctvRawImg) elements.cctvRawImg.style.display = "none";
          if (elements.cctvRawEmpty) elements.cctvRawEmpty.style.display = "none";
          if (elements.cctvRawMeta) elements.cctvRawMeta.textContent = "● LIVE CAM-01 STREAMING";
          if (elements.cctvRawFpsTag) elements.cctvRawFpsTag.textContent = "30 FPS · OPTICAL LIVE";
        }

        elements.btnToggleWebcam.textContent = "Disconnect Camera";
        elements.btnCaptureWebcam.disabled = false;
        if (elements.cameraLensBadge) {
          elements.cameraLensBadge.textContent = "📷 DEFAULT SYSTEM CAMERA";
        }
        logAudit("Hardware video device connected with default lens.");
      } catch (fallbackErr) {
        alert("Camera stream unavailable. On mobile browsers over HTTP, tap 'Take Photo with Back Camera' below for native rear camera capture.");
        logAudit("Camera access error: " + err.message);
      }
    }
  }

  function stopWebcam() {
    if (state.isLiveStreaming) {
      toggleLiveAiStream();
    }
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach((t) => t.stop());
      state.webcamStream = null;
    }
    elements.webcamVideo.srcObject = null;
    if (elements.cctvRawVideo) {
      elements.cctvRawVideo.srcObject = null;
      elements.cctvRawVideo.style.display = "none";
      if (elements.cctvRawEmpty) elements.cctvRawEmpty.style.display = "flex";
      if (elements.cctvRawMeta) elements.cctvRawMeta.textContent = "FEED STANDBY";
    }
    elements.btnToggleWebcam.textContent = "Connect Camera";
    elements.btnCaptureWebcam.disabled = true;
    logAudit("Hardware video device disconnected.");
  }

  // Camera Flip / Switch Button (Front/Back)
  if (elements.btnSwitchCamera) {
    elements.btnSwitchCamera.addEventListener("click", async () => {
      state.cameraFacingMode = state.cameraFacingMode === "environment" ? "user" : "environment";
      if (state.webcamStream) {
        stopWebcam();
        await startWebcam();
      } else {
        if (elements.cameraLensBadge) {
          elements.cameraLensBadge.textContent = state.cameraFacingMode === "environment"
            ? "📷 BACK CAMERA (SELECTED)"
            : "🤳 FRONT CAMERA (SELECTED)";
        }
        logAudit(`Camera set to ${state.cameraFacingMode === "environment" ? "back" : "front"} lens.`);
      }
    });
  }

  // Native Mobile Back Camera Trigger (works universally on all mobile browsers)
  if (elements.mobileCameraInput) {
    elements.mobileCameraInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        const photo = e.target.files[0];
        handleSelectedFile(photo);
        switchTab("upload");
        logAudit(`Mobile rear camera snapshot staged: ${photo.name} (${formatBytes(photo.size)})`);
      }
    });
  }

  elements.btnCaptureWebcam.addEventListener("click", () => {
    if (!elements.webcamVideo.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = elements.webcamVideo.videoWidth;
    canvas.height = elements.webcamVideo.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(elements.webcamVideo, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      const snapshotFile = new File([blob], `cctv_capture_${Date.now()}.jpg`, {
        type: "image/jpeg",
      });
      state.selectedFile = snapshotFile;
      state.selectedSampleUrl = null;
      switchTab("upload");
      handleSelectedFile(snapshotFile);
      logAudit("Live frame snapshot captured.");
    }, "image/jpeg");
  });

  /* ==========================================================
     7. SENSITIVITY CONTROLS & VALIDATION
     ========================================================== */
  elements.confSlider.addEventListener("input", (e) => {
    elements.confValue.textContent = `${Math.round(e.target.value * 100)}%`;
  });

  function validateProcessButton() {
    const hasUpload = state.activeTab === "upload" && state.selectedFile !== null;
    const hasSample = state.activeTab === "samples" && state.selectedSampleUrl !== null;
    elements.btnProcess.disabled = !hasUpload && !hasSample;
  }

  /* ==========================================================
     8. INFERENCE PIPELINE EXECUTION
     ========================================================== */
  elements.btnProcess.addEventListener("click", async () => {
    if (state.isProcessing) return;

    let uploadPayload = null;

    if (state.activeTab === "upload" && state.selectedFile) {
      uploadPayload = state.selectedFile;
    } else if (state.activeTab === "samples" && state.selectedSampleUrl) {
      try {
        logAudit("Fetching selected sample asset...");
        const res = await fetch(state.selectedSampleUrl);
        const blob = await res.blob();
        const filename = state.selectedSampleUrl.split("/").pop() || "sample.jpg";
        uploadPayload = new File([blob], filename, { type: blob.type || "image/jpeg" });
      } catch (err) {
        alert("Failed to load sample image for inference.");
        return;
      }
    }

    if (!uploadPayload) return;

    const isVideo = uploadPayload.type.startsWith("video") || uploadPayload.name.match(/\.(mp4|avi|mov|webm|mkv|3gp)$/i);
    setProcessingUI(true, isVideo);
    const startTime = performance.now();

    let stageTimer = setInterval(() => {
      const elapsedSec = Math.floor((performance.now() - startTime) / 1000);
      const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      if (elements.scannerElapsed) {
        elements.scannerElapsed.textContent = `${m}:${s}s elapsed`;
      }

      if (elements.scannerStatus) {
        if (elapsedSec < 3) {
          elements.scannerStatus.textContent = "Uploading media to local neural server...";
        } else if (elapsedSec < 7) {
          elements.scannerStatus.textContent = isVideo
            ? "Decoding video stream & running YOLOv8 tracking..."
            : "Running vehicle detection & plate localization...";
          if (elements.scannerInnerIcon) elements.scannerInnerIcon.textContent = "🔍";
        } else if (elapsedSec < 13) {
          elements.scannerStatus.textContent = "Reading number plates with EasyOCR + TrOCR brain...";
          if (elements.scannerInnerIcon) elements.scannerInnerIcon.textContent = "🧠";
        } else {
          elements.scannerStatus.textContent = "Verifying state jurisdiction & consensus filtering...";
          if (elements.scannerInnerIcon) elements.scannerInnerIcon.textContent = "📋";
        }
      }
    }, 1000);

    const formData = new FormData();
    formData.append("file", uploadPayload);

    const confVal = parseFloat(elements.confSlider.value);
    const endpoint = `${API_BASE}/analyze-parking?conf=${confVal}`;

    try {
      logAudit(`Initiating neural pipeline for ${uploadPayload.name}...`);
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      clearInterval(stageTimer);

      const rawText = await response.text();
      let result = null;
      try {
        result = JSON.parse(rawText);
      } catch (_) {
        let parsedErr = `HTTP ${response.status}`;
        if (rawText && rawText.includes('<title>')) {
          const m = rawText.match(/<title>(.*?)<\/title>/i);
          if (m && m[1]) parsedErr = m[1].trim();
        } else if (rawText) {
          parsedErr = rawText.slice(0, 160).trim();
        }
        throw new Error(parsedErr);
      }

      if (!response.ok) {
        throw new Error((result && result.error) ? result.error : `HTTP ${response.status}`);
      }

      if (result && result.status === "error") {
        throw new Error(result.error || "Server reported analysis error.");
      }

      const executionDuration = Math.round(performance.now() - startTime);

      // Flash success on overlay before closing
      if (elements.scannerCircleSpinner) {
        elements.scannerCircleSpinner.style.stroke = "var(--status-ok)";
      }
      if (elements.scannerInnerIcon) elements.scannerInnerIcon.textContent = "✅";
      if (elements.scannerStatus) {
        elements.scannerStatus.textContent = "Analysis complete! Rendering results...";
        elements.scannerStatus.style.color = "var(--status-ok)";
      }

      setTimeout(() => {
        setProcessingUI(false);
        renderInferenceResults(result, uploadPayload.name, executionDuration);
        logAudit(
          `Analysis completed in ${executionDuration}ms: ${result.metrics.total_vehicles} vehicles, ${result.number_plates.length} plates logged.`
        );
      }, 450);

    } catch (error) {
      clearInterval(stageTimer);
      showScannerError(error.message);
      logAudit("Error during inference: " + error.message);
    }
  });

  function setProcessingUI(active, isVideo = false) {
    state.isProcessing = active;
    elements.btnProcess.disabled = active;
    elements.scannerOverlay.classList.toggle("active", active);

    if (active) {
      if (elements.scannerErrorBox) elements.scannerErrorBox.style.display = "none";
      if (elements.scannerCircleSpinner) {
        elements.scannerCircleSpinner.style.stroke = "var(--accent-primary)";
        elements.scannerCircleSpinner.style.animation = "spinCircle 1.6s cubic-bezier(0.4, 0, 0.2, 1) infinite";
        elements.scannerCircleSpinner.style.strokeDashoffset = "200";
      }
      if (elements.scannerInnerIcon) elements.scannerInnerIcon.textContent = isVideo ? "🎥" : "⚡";
      if (elements.scannerTitle) {
        elements.scannerTitle.textContent = isVideo ? "ANALYZING VIDEO FOOTAGE" : "ANALYZING CAPTURE";
        elements.scannerTitle.style.color = "var(--text-main)";
      }
      if (elements.scannerStatus) {
        elements.scannerStatus.textContent = "Uploading media to local neural server...";
        elements.scannerStatus.style.color = "var(--accent-primary)";
      }
      if (elements.scannerElapsed) elements.scannerElapsed.textContent = "00:00s elapsed";
      if (elements.scannerTimerDot) {
        elements.scannerTimerDot.style.background = "var(--status-ok)";
        elements.scannerTimerDot.style.boxShadow = "0 0 6px var(--status-ok)";
      }
      if (elements.scannerServerBadge) {
        elements.scannerServerBadge.textContent = "SERVER ACTIVE";
        elements.scannerServerBadge.style.color = "var(--status-ok)";
        elements.scannerServerBadge.style.background = "var(--status-ok-bg)";
        elements.scannerServerBadge.style.borderColor = "var(--status-ok-border)";
      }
      if (elements.scannerReassurance) {
        elements.scannerReassurance.style.display = "block";
        elements.scannerReassurance.textContent = isVideo
          ? "Please wait. Video frames are being decoded and tracked on CPU with dual OCR."
          : "Please wait. Image received and local models are processing.";
      }
      elements.btnProcess.innerHTML = `<span class="spinner" style="width: 16px; height: 16px; border-width: 2px;"></span> Processing...`;
    } else {
      elements.btnProcess.innerHTML = `<span>▶ Run Analysis</span>`;
      validateProcessButton();
    }
  }

  function showScannerError(errorMessage) {
    if (elements.scannerCircleSpinner) {
      elements.scannerCircleSpinner.style.stroke = "var(--status-danger)";
      elements.scannerCircleSpinner.style.animation = "none";
      elements.scannerCircleSpinner.style.strokeDashoffset = "0";
    }
    if (elements.scannerInnerIcon) elements.scannerInnerIcon.textContent = "❌";
    if (elements.scannerTitle) {
      elements.scannerTitle.textContent = "ANALYSIS ENCOUNTERED AN ERROR";
      elements.scannerTitle.style.color = "var(--status-danger)";
    }
    if (elements.scannerStatus) {
      elements.scannerStatus.textContent = "The model could not finish analyzing the file.";
      elements.scannerStatus.style.color = "var(--status-danger)";
    }
    if (elements.scannerServerBadge) {
      elements.scannerServerBadge.textContent = "ERROR STOP";
      elements.scannerServerBadge.style.color = "var(--status-danger)";
      elements.scannerServerBadge.style.background = "var(--status-danger-bg)";
      elements.scannerServerBadge.style.borderColor = "var(--status-danger-border)";
    }
    if (elements.scannerTimerDot) {
      elements.scannerTimerDot.style.background = "var(--status-danger)";
      elements.scannerTimerDot.style.boxShadow = "0 0 6px var(--status-danger)";
    }
    if (elements.scannerReassurance) elements.scannerReassurance.style.display = "none";
    if (elements.scannerErrorBox) elements.scannerErrorBox.style.display = "block";
    if (elements.scannerErrorMessage) {
      elements.scannerErrorMessage.textContent = errorMessage || "Unknown server or network error.";
    }
    elements.btnProcess.innerHTML = `<span>▶ Run Analysis</span>`;
    validateProcessButton();
  }

  if (elements.btnDismissScannerError) {
    elements.btnDismissScannerError.addEventListener("click", () => {
      setProcessingUI(false);
    });
  }

  /* ==========================================================
     9. RENDER INFERENCE RESULTS & TELEMETRY
     ========================================================== */
  function renderInferenceResults(data, filename, durationMs) {
    state.lastResult = data;
    state.detectedPlates = data.number_plates || [];

    const { metrics, actionable_alerts, media_type, annotated_media } = data;

    // 1. Update Metrics Cards (Cars, Trucks, Two-wheelers / Bikes, Buses)
    elements.statTotalVehicles.textContent = metrics.total_vehicles ?? 0;
    elements.statLightVehicles.textContent = (metrics.cars_and_trucks || 0) + (metrics.two_wheelers || 0);
    elements.statHeavyBuses.textContent = metrics.buses ?? 0;
    elements.statPlatesCount.textContent = state.detectedPlates.length;

    // Alert styling on bus counter
    if (metrics.buses > 0) {
      elements.statHeavyBuses.style.color = "var(--status-danger)";
    } else {
      elements.statHeavyBuses.style.color = "var(--text-main)";
    }

    // 2. Update Directives & Capacity
    if (actionable_alerts.kitchen && actionable_alerts.kitchen.includes("URGENT")) {
      elements.alertKitchen.className = "alert-card urgent";
      elements.alertKitchenText.textContent = actionable_alerts.kitchen;
    } else {
      elements.alertKitchen.className = "alert-card normal";
      elements.alertKitchenText.textContent = actionable_alerts.kitchen || "Standard Prep Cadence";
    }

    if (actionable_alerts.floor_manager && actionable_alerts.floor_manager.includes("Reallocate")) {
      elements.alertFloor.className = "alert-card urgent";
      elements.alertFloorText.textContent = actionable_alerts.floor_manager;
    } else {
      elements.alertFloor.className = "alert-card normal";
      elements.alertFloorText.textContent = actionable_alerts.floor_manager || "Standard Floor Coverage";
    }

    // Parking Capacity Fill (Benchmark: 20 bays)
    const maxCapacity = 20;
    const capacityPct = Math.min(100, Math.round((metrics.total_vehicles / maxCapacity) * 100));
    elements.capacityFill.style.width = `${capacityPct}%`;
    elements.capacityText.textContent = `${capacityPct}% Occupancy (${metrics.total_vehicles}/${maxCapacity})`;
    if (capacityPct >= 85) {
      elements.capacityFill.style.backgroundColor = "var(--status-danger)";
    } else if (capacityPct >= 60) {
      elements.capacityFill.style.backgroundColor = "var(--status-warn)";
    } else {
      elements.capacityFill.style.backgroundColor = "var(--status-ok)";
    }

    // 3. Render Media Viewport & Synchronize Dual CCTV Feeds
    if (elements.viewportEmpty) elements.viewportEmpty.style.display = "none";
    resetZoom();

    const mediaSrc = annotated_media.startsWith('http')
      ? annotated_media
      : `data:${media_type === "video" ? "video/mp4" : "image/jpeg"};base64,${annotated_media}`;

    if (media_type === "video") {
      if (elements.viewportImg) elements.viewportImg.style.display = "none";
      if (elements.viewportVideo) {
        elements.viewportVideo.style.display = "block";
        elements.viewportVideo.src = mediaSrc;

        // Synchronize Raw CCTV video with AI Inference video
        if (elements.cctvRawVideo && elements.cctvRawVideo.src) {
          elements.cctvRawVideo.currentTime = 0;
          elements.viewportVideo.currentTime = 0;

          elements.cctvRawVideo.onplay = () => {
            if (elements.viewportVideo.paused) elements.viewportVideo.play().catch(() => {});
          };
          elements.cctvRawVideo.onpause = () => {
            if (!elements.viewportVideo.paused) elements.viewportVideo.pause();
          };
          elements.cctvRawVideo.onseeked = () => {
            if (Math.abs(elements.viewportVideo.currentTime - elements.cctvRawVideo.currentTime) > 0.15) {
              elements.viewportVideo.currentTime = elements.cctvRawVideo.currentTime;
            }
          };

          elements.viewportVideo.onplay = () => {
            if (elements.cctvRawVideo.paused) elements.cctvRawVideo.play().catch(() => {});
          };
          elements.viewportVideo.onpause = () => {
            if (!elements.cctvRawVideo.paused) elements.cctvRawVideo.pause();
          };
          elements.viewportVideo.onseeked = () => {
            if (Math.abs(elements.cctvRawVideo.currentTime - elements.viewportVideo.currentTime) > 0.15) {
              elements.cctvRawVideo.currentTime = elements.viewportVideo.currentTime;
            }
          };

          elements.cctvRawVideo.play().catch(() => {});
        }
        elements.viewportVideo.play().catch(() => {});
      }
    } else {
      if (elements.viewportVideo) elements.viewportVideo.style.display = "none";
      if (elements.viewportImg) {
        elements.viewportImg.style.display = "block";
        elements.viewportImg.src = mediaSrc;
      }
    }

    elements.viewportMetaInfo.textContent = `SRC: ${filename} | LATENCY: ${durationMs}ms | DETECTIONS: ${metrics.total_vehicles}`;
    elements.btnDownloadMedia.disabled = false;

    // 4. Render License Plate Ledger
    renderPlateLedger(state.detectedPlates);
  }

  /* ==========================================================
     10. LICENSE PLATE LEDGER & SEARCH
     ========================================================== */
  function renderPlateLedger(plates) {
    elements.ledgerCountBadge.textContent = `${plates.length} LOGGED`;
    elements.ledgerList.innerHTML = "";

    if (!plates || plates.length === 0) {
      elements.ledgerList.innerHTML = `
        <div class="ledger-empty">
          <div class="ledger-empty-icon">📋</div>
          <div>No license plates identified in current feed frame.</div>
        </div>
      `;
      return;
    }

    plates.forEach((plate, index) => {
      const card = document.createElement("div");
      card.className = "plate-row-card";
      const plateText = (plate.text || "").toUpperCase();
      const stateCode = (plate.state_code || "IND").toUpperCase();
      const stateName = plate.state_name || "Standard Registration";
      
      // Store plate number, state code, and state name for search filtering
      card.dataset.plate = `${plateText} ${stateCode} ${stateName.toUpperCase()}`;

      const timeLogged = new Date().toLocaleTimeString("en-GB", { hour12: false });
      const cropSrc = plate.chunk ? (plate.chunk.startsWith("http") ? plate.chunk : `data:image/jpeg;base64,${plate.chunk}`) : "";
      const isVerified = plate.is_valid_format !== false;

      card.innerHTML = `
        ${cropSrc ? `<img src="${cropSrc}" alt="Plate Crop" class="plate-crop-thumb">` : ""}
        <div class="plate-row-content">
          <span class="plate-number-badge">${escapeHtml(plateText)}</span>
          <div class="plate-state-pill" title="Vehicle State: ${escapeHtml(stateName)}">
            <span class="state-code-chip">${escapeHtml(stateCode)}</span>
            <span class="state-name-text">${escapeHtml(stateName)}</span>
          </div>
          <div class="plate-row-meta">
            <span class="${isVerified ? 'badge-verified' : 'badge-identified'}">
              ${isVerified ? '✓ VERIFIED RTO' : 'IDENTIFIED'}
            </span>
            <span>•</span>
            <span>${timeLogged}</span>
          </div>
        </div>
        <button class="btn-copy-plate" title="Copy Registration Number">Copy</button>
      `;

      const copyBtn = card.querySelector(".btn-copy-plate");
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(plateText);
        copyBtn.textContent = "Copied!";
        copyBtn.style.color = "var(--status-ok)";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
          copyBtn.style.color = "";
        }, 1500);
      });

      elements.ledgerList.appendChild(card);
    });
  }

  // Filter plates in ledger
  elements.plateSearchInput.addEventListener("input", (e) => {
    const query = e.target.value.trim().toUpperCase();
    const rows = elements.ledgerList.querySelectorAll(".plate-row-card");
    rows.forEach((row) => {
      const plateText = row.dataset.plate || "";
      row.style.display = plateText.includes(query) ? "flex" : "none";
    });
  });

  /* ==========================================================
     11. VIEWPORT CONTROLS (ZOOM / PAN / DOWNLOAD)
     ========================================================== */
  elements.btnZoomIn.addEventListener("click", () => {
    state.zoomLevel = Math.min(3.0, state.zoomLevel + 0.25);
    applyZoom();
  });

  elements.btnZoomOut.addEventListener("click", () => {
    state.zoomLevel = Math.max(0.75, state.zoomLevel - 0.25);
    applyZoom();
  });

  elements.btnZoomReset.addEventListener("click", resetZoom);

  function resetZoom() {
    state.zoomLevel = 1.0;
    applyZoom();
  }

  function applyZoom() {
    elements.viewportImg.style.transform = `scale(${state.zoomLevel})`;
  }

  elements.btnFullscreen.addEventListener("click", () => {
    if (!document.fullscreenElement) {
      elements.viewportContainer.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  elements.btnDownloadMedia.addEventListener("click", () => {
    if (!state.lastResult || !state.lastResult.annotated_media) return;
    const a = document.createElement("a");
    const isVideo = state.lastResult.media_type === "video";
    const media = state.lastResult.annotated_media;
    if (media.startsWith("http")) {
      a.href = media;
      a.target = "_blank";
    } else {
      a.href = `data:${isVideo ? "video/mp4" : "image/jpeg"};base64,${media}`;
    }
    a.download = `mothers_hut_annotated_${Date.now()}.${isVideo ? "mp4" : "jpg"}`;
    a.click();
    logAudit("Annotated media export downloaded.");
  });

  /* ==========================================================
     12. DATA EXPORT (SUPABASE DATABASE CSV & RECOVERY)
     ========================================================== */
  elements.btnExportCsv.addEventListener("click", () => {
    const selectedDate = elements.ledgerDateSelect ? elements.ledgerDateSelect.value : "today";
    const actualDateParam = selectedDate === "today" ? new Date().toISOString().slice(0, 10) : selectedDate;
    
    // Direct download from Supabase backend database (Available even days later)
    const downloadUrl = `/api/ledger/export-csv?date=${encodeURIComponent(actualDateParam)}`;
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `mothers_hut_ledger_${actualDateParam}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    logAudit(`Supabase Ledger CSV exported for date: ${actualDateParam}`);
  });

  async function loadAvailableLedgerDates() {
    if (!elements.ledgerDateSelect) return;
    try {
      const res = await fetch("/api/ledger/dates");
      const data = await res.json();
      if (data && data.dates && data.dates.length > 0) {
        const todayIso = new Date().toISOString().slice(0, 10);
        let opts = `<option value="today">Today (${todayIso})</option>`;
        opts += `<option value="all">All Dates (${data.dates.reduce((acc, d) => acc + d.plate_count, 0)} plates)</option>`;
        data.dates.forEach(d => {
          opts += `<option value="${d.date}">${d.date} (${d.plate_count} plates)</option>`;
        });
        elements.ledgerDateSelect.innerHTML = opts;
      }
    } catch (e) {
      console.warn("Could not load ledger dates:", e);
    }
  }

  async function loadPersistentLedger(dateVal = null) {
    try {
      const todayIso = new Date().toISOString().slice(0, 10);
      const queryDate = (!dateVal || dateVal === "today") ? todayIso : dateVal;
      const url = `/api/ledger/entries?date=${encodeURIComponent(queryDate)}`;
      const res = await fetch(url);
      const data = await res.json();
      if (data && data.entries && data.entries.length > 0) {
        state.detectedPlates = data.entries.map(e => ({
          text: e.plate_text,
          state_code: e.state_code,
          state_name: e.state_name,
          is_valid_format: e.is_valid_format,
          confidence_score: e.confidence_score,
          timestamp: e.timestamp,
          chunk: ""
        }));
        renderLedger(state.detectedPlates);
        logAudit(`Loaded ${data.entries.length} persistent plate records from Supabase (${queryDate}).`);
      } else if (dateVal && dateVal !== "today") {
        state.detectedPlates = [];
        renderLedger([]);
        logAudit(`No plate records found in Supabase for ${queryDate}.`);
      }
    } catch (e) {
      console.warn("Could not load persistent ledger from Supabase:", e);
    }
  }

  if (elements.ledgerDateSelect) {
    elements.ledgerDateSelect.addEventListener("change", (e) => {
      loadPersistentLedger(e.target.value);
    });
  }

  elements.btnCopyAllPlates.addEventListener("click", () => {
    if (!state.detectedPlates || state.detectedPlates.length === 0) return;
    const text = state.detectedPlates.map((p) => p.text).join("\n");
    navigator.clipboard.writeText(text);
    elements.btnCopyAllPlates.textContent = "Copied All!";
    setTimeout(() => {
      elements.btnCopyAllPlates.textContent = "Copy All";
    }, 1500);
    logAudit("Copied plate ledger to clipboard.");
  });

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Run Startup
  initClock();
  initTheme();
  initCctvConsole();
  checkBackendHealth();
  loadAvailableLedgerDates();
  loadPersistentLedger();
  setInterval(checkBackendHealth, 30000);
});

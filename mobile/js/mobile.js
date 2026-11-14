/**
 * Mother's Hut - Mobile Gate Operations Console
 * Dedicated Mobile Controller
 * - Real-time Back Camera (Environment) Stream
 * - Live Continuous Video Scanning Loop
 * - MediaRecorder Live Video Clip Capture
 * - Native Mobile Hardware Back-Camera Shutter & Video Recorder
 * - State Jurisdiction & License Plate Ledger
 */

(function() {
  'use strict';

  // API base URL auto-detection
  const API_BASE = (window.location.origin && window.location.origin.startsWith('http')) 
    ? window.location.origin 
    : 'http://127.0.0.1:8000';

  // DOM Elements
  const mobileLiveVideo = document.getElementById('mobileLiveVideo');
  const cameraPlaceholder = document.getElementById('cameraPlaceholder');
  const scannerOverlay = document.getElementById('scannerOverlay');
  const lensIndicatorText = document.getElementById('lensIndicatorText');
  const btnToggleLiveCam = document.getElementById('btnToggleLiveCam');
  const btnFlipCam = document.getElementById('btnFlipCam');
  const btnSnapLiveFrame = document.getElementById('btnSnapLiveFrame');
  const btnRecordClip = document.getElementById('btnRecordClip');
  const btnLiveScanToggle = document.getElementById('btnLiveScanToggle');
  
  const mobileVideoRecordInput = document.getElementById('mobileVideoRecordInput');
  const mobilePhotoInput = document.getElementById('mobilePhotoInput');
  const fileUploadInput = document.getElementById('fileUploadInput');

  const healthDot = document.getElementById('healthDot');
  const healthLabel = document.getElementById('healthLabel');
  const hardwareBadge = document.getElementById('hardwareBadge');
  const themeToggleMobile = document.getElementById('themeToggleMobile');
  const mobileToast = document.getElementById('mobileToast');

  // Metrics & Alerts
  const mTotalVehicles = document.getElementById('mTotalVehicles');
  const mTotalCars = document.getElementById('mTotalCars');
  const mTotalBuses = document.getElementById('mTotalBuses');
  const mKitchenAlert = document.getElementById('mKitchenAlert');
  const mKitchenCard = document.getElementById('mKitchenCard');
  const mFloorAlert = document.getElementById('mFloorAlert');

  // Results
  const resultCard = document.getElementById('resultCard');
  const resultMediaWrap = document.getElementById('resultMediaWrap');
  const resultInfoTag = document.getElementById('resultInfoTag');

  // Ledger
  const mobilePlateList = document.getElementById('mobilePlateList');
  const searchPlatesMobile = document.getElementById('searchPlatesMobile');
  const btnExportCsvMobile = document.getElementById('btnExportCsvMobile');

  // State Management
  let mediaStream = null;
  let currentFacingMode = 'environment'; // Default to Back Camera
  let isScanningLive = false;
  let liveScanInterval = null;
  let isProcessingFrame = false;
  let mediaRecorder = null;
  let recordedChunks = [];
  let isRecordingClip = false;
  let allLedgerPlates = [];

  // -------------------------------------------------------------
  // Toast Helper
  // -------------------------------------------------------------
  function showToast(message, duration = 3000) {
    if (!mobileToast) return;
    mobileToast.textContent = message;
    mobileToast.classList.add('show');
    setTimeout(() => {
      mobileToast.classList.remove('show');
    }, duration);
  }

  // -------------------------------------------------------------
  // Theme Switching (Dark / Light)
  // -------------------------------------------------------------
  function initTheme() {
    const savedTheme = localStorage.getItem('mh_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeButton(savedTheme);

    if (themeToggleMobile) {
      themeToggleMobile.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('mh_theme', next);
        updateThemeButton(next);
      });
    }
  }

  function updateThemeButton(theme) {
    if (!themeToggleMobile) return;
    themeToggleMobile.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
  }

  // -------------------------------------------------------------
  // Navigation Tabs
  // -------------------------------------------------------------
  function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-tab-btn');
    const tabPanels = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        navButtons.forEach(b => b.classList.remove('active'));
        tabPanels.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const activePanel = document.getElementById(`tab-${targetTab}`);
        if (activePanel) activePanel.classList.add('active');
      });
    });
  }

  // -------------------------------------------------------------
  // Health & Service Telemetry
  // -------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        if (healthDot) {
          healthDot.classList.remove('connecting');
          healthDot.style.backgroundColor = 'var(--status-ok)';
        }
        if (healthLabel) healthLabel.textContent = 'ONLINE (LOCAL/FREE)';
        if (hardwareBadge && data.device) {
          hardwareBadge.textContent = data.device.toUpperCase();
        }
      } else {
        throw new Error('Non-200');
      }
    } catch (e) {
      if (healthDot) {
        healthDot.classList.add('connecting');
        healthDot.style.backgroundColor = 'var(--status-warn)';
      }
      if (healthLabel) healthLabel.textContent = 'CONNECTING...';
    }
  }

  // -------------------------------------------------------------
  // Live Back Camera Management
  // -------------------------------------------------------------
  async function startLiveCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('Live WebRTC requires HTTPS or Chrome Flag. Use Native Shutter buttons below!');
      document.getElementById('httpCameraNotice').style.display = 'block';
      return false;
    }

    stopLiveCamera();

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: currentFacingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    };

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      mobileLiveVideo.srcObject = mediaStream;
      await mobileLiveVideo.play();

      cameraPlaceholder.style.display = 'none';
      scannerOverlay.style.display = 'flex';
      btnToggleLiveCam.textContent = 'Stop Feed';
      btnToggleLiveCam.style.background = 'var(--status-danger-bg)';
      btnToggleLiveCam.style.color = 'var(--status-danger)';
      
      btnSnapLiveFrame.disabled = false;
      btnRecordClip.disabled = false;
      btnLiveScanToggle.disabled = false;

      // Update badge
      const activeTrack = mediaStream.getVideoTracks()[0];
      const trackLabel = activeTrack ? activeTrack.label.toLowerCase() : '';
      const isBack = currentFacingMode === 'environment' || trackLabel.includes('back') || trackLabel.includes('rear');
      lensIndicatorText.textContent = isBack ? 'LIVE REAR CAMERA' : 'LIVE FRONT CAMERA';
      
      showToast(`Camera feed active (${isBack ? 'Rear' : 'Front'})`);
      return true;
    } catch (err) {
      console.warn('getUserMedia error:', err);
      cameraPlaceholder.style.display = 'flex';
      scannerOverlay.style.display = 'none';
      showToast('Camera access denied or unverified. Use Native Camera buttons below!');
      document.getElementById('httpCameraNotice').style.display = 'block';
      return false;
    }
  }

  function stopLiveCamera() {
    stopLiveScannerLoop();
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    if (mobileLiveVideo) mobileLiveVideo.srcObject = null;
    if (cameraPlaceholder) cameraPlaceholder.style.display = 'flex';
    if (scannerOverlay) scannerOverlay.style.display = 'none';
    if (btnToggleLiveCam) {
      btnToggleLiveCam.textContent = 'Start Live Camera';
      btnToggleLiveCam.style.background = '';
      btnToggleLiveCam.style.color = '';
    }
    if (btnSnapLiveFrame) btnSnapLiveFrame.disabled = true;
    if (btnRecordClip) btnRecordClip.disabled = true;
    if (btnLiveScanToggle) {
      btnLiveScanToggle.disabled = true;
      btnLiveScanToggle.textContent = '⚡ Auto-Scan Stream: OFF';
      btnLiveScanToggle.classList.remove('active');
    }
  }

  // Camera Flip Toggle
  async function flipCamera() {
    currentFacingMode = (currentFacingMode === 'environment') ? 'user' : 'environment';
    if (mediaStream) {
      await startLiveCamera();
    } else {
      showToast(`Default lens set to ${currentFacingMode === 'environment' ? 'Rear' : 'Front'}`);
    }
  }

  // -------------------------------------------------------------
  // Live Continuous Stream Scanning Loop
  // -------------------------------------------------------------
  function toggleLiveScanner() {
    if (isScanningLive) {
      stopLiveScannerLoop();
      showToast('Live auto-scan paused');
    } else {
      if (!mediaStream) {
        startLiveCamera().then(ok => {
          if (ok) startLiveScannerLoop();
        });
      } else {
        startLiveScannerLoop();
      }
    }
  }

  function startLiveScannerLoop() {
    isScanningLive = true;
    btnLiveScanToggle.textContent = '🟢 Auto-Scan Stream: ACTIVE';
    btnLiveScanToggle.classList.add('active');
    showToast('Continuous live stream analysis active');

    liveScanInterval = setInterval(async () => {
      if (isProcessingFrame || !mediaStream || mobileLiveVideo.readyState < 2) return;
      isProcessingFrame = true;
      try {
        const blob = await captureVideoFrameBlob();
        if (blob) {
          await sendInferenceRequest(blob, 'live_stream_frame.jpg', false);
        }
      } catch (err) {
        console.warn('Live scan cycle failed:', err);
      } finally {
        isProcessingFrame = false;
      }
    }, 1200);
  }

  function stopLiveScannerLoop() {
    isScanningLive = false;
    if (liveScanInterval) {
      clearInterval(liveScanInterval);
      liveScanInterval = null;
    }
    if (btnLiveScanToggle) {
      btnLiveScanToggle.textContent = '⚡ Auto-Scan Stream: OFF';
      btnLiveScanToggle.classList.remove('active');
    }
  }

  // Capture single frame from <video> as Blob
  function captureVideoFrameBlob() {
    return new Promise(resolve => {
      if (!mobileLiveVideo || mobileLiveVideo.videoWidth === 0) return resolve(null);
      const canvas = document.createElement('canvas');
      canvas.width = mobileLiveVideo.videoWidth;
      canvas.height = mobileLiveVideo.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(mobileLiveVideo, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.88);
    });
  }

  // -------------------------------------------------------------
  // Live Video Clip Recording (MediaRecorder)
  // -------------------------------------------------------------
  function recordLiveClip(seconds = 6) {
    if (!mediaStream) {
      showToast('Please start the live camera first.');
      return;
    }

    if (isRecordingClip) return;
    isRecordingClip = true;
    recordedChunks = [];

    let mimeType = 'video/webm';
    if (!MediaRecorder.isTypeSupported('video/webm')) {
      if (MediaRecorder.isTypeSupported('video/mp4')) {
        mimeType = 'video/mp4';
      }
    }

    try {
      mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
    } catch (e) {
      mediaRecorder = new MediaRecorder(mediaStream);
    }

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
      const clipBlob = new Blob(recordedChunks, { type: mimeType });
      btnRecordClip.classList.remove('recording');
      btnRecordClip.textContent = '🎥 Record Live Clip (5s)';
      isRecordingClip = false;
      showToast('Processing recorded live video clip...');
      await sendInferenceRequest(clipBlob, `live_clip.${ext}`, true);
    };

    mediaRecorder.start();
    btnRecordClip.classList.add('recording');
    showToast(`Recording live back-camera video (${seconds}s)...`);

    let remaining = seconds;
    btnRecordClip.textContent = `🔴 Recording (${remaining}s)...`;
    const countdown = setInterval(() => {
      remaining--;
      if (remaining > 0) {
        btnRecordClip.textContent = `🔴 Recording (${remaining}s)...`;
      } else {
        clearInterval(countdown);
        if (mediaRecorder && mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }
    }, 1000);
  }

  // Processing Circle Modal Elements
  const processingModal = document.getElementById('processingModal');
  const processingCircleSpinner = document.getElementById('processingCircleSpinner');
  const processingInnerIcon = document.getElementById('processingInnerIcon');
  const processingTitle = document.getElementById('processingTitle');
  const processingStatus = document.getElementById('processingStatus');
  const processingTimer = document.getElementById('processingTimer');
  const processingTimerDot = document.getElementById('processingTimerDot');
  const processingServerBadge = document.getElementById('processingServerBadge');
  const processingReassurance = document.getElementById('processingReassurance');
  const processingErrorBox = document.getElementById('processingErrorBox');
  const processingErrorMessage = document.getElementById('processingErrorMessage');
  const btnDismissError = document.getElementById('btnDismissError');

  let processingInterval = null;
  let processingStartTime = 0;

  function showProcessingOverlay(filename = 'media', isVideo = false) {
    if (!processingModal) return;
    processingStartTime = Date.now();
    
    // Reset modal state
    if (processingErrorBox) processingErrorBox.style.display = 'none';
    if (processingCircleSpinner) {
      processingCircleSpinner.style.stroke = 'var(--accent-primary)';
      processingCircleSpinner.style.animation = 'spinCircle 1.6s cubic-bezier(0.4, 0, 0.2, 1) infinite';
      processingCircleSpinner.style.strokeDashoffset = '200';
    }
    if (processingInnerIcon) {
      processingInnerIcon.textContent = isVideo ? '🎥' : '⚡';
    }
    if (processingTitle) {
      processingTitle.textContent = isVideo ? 'ANALYZING VIDEO FOOTAGE' : 'ANALYZING CAPTURE';
      processingTitle.style.color = 'var(--text-main)';
    }
    if (processingStatus) {
      processingStatus.textContent = 'Uploading media to local neural server...';
      processingStatus.style.color = 'var(--accent-primary)';
    }
    if (processingTimer) processingTimer.textContent = '00:00s elapsed';
    if (processingTimerDot) {
      processingTimerDot.style.background = 'var(--status-ok)';
      processingTimerDot.style.boxShadow = '0 0 6px var(--status-ok)';
    }
    if (processingServerBadge) {
      processingServerBadge.textContent = 'SERVER ACTIVE';
      processingServerBadge.style.color = 'var(--status-ok)';
      processingServerBadge.style.borderColor = 'var(--status-ok-border)';
      processingServerBadge.style.background = 'var(--status-ok-bg)';
    }
    if (processingReassurance) {
      processingReassurance.style.display = 'block';
      processingReassurance.textContent = isVideo
        ? 'Please wait. Your phone video is being decoded frame-by-frame on CPU with vehicle tracking and dual OCR.'
        : 'Please wait. Your image has been received and local models are processing.';
    }

    processingModal.style.display = 'flex';
    // Trigger transition
    setTimeout(() => processingModal.classList.add('active'), 10);

    // Dynamic timer & stage messages
    if (processingInterval) clearInterval(processingInterval);
    processingInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - processingStartTime) / 1000);
      const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
      const s = String(elapsedSec % 60).padStart(2, '0');
      if (processingTimer) processingTimer.textContent = `${m}:${s}s elapsed`;

      // Dynamic sub-status messages as time progresses
      if (processingStatus) {
        if (elapsedSec < 3) {
          processingStatus.textContent = 'Uploading media to local neural server...';
        } else if (elapsedSec < 7) {
          processingStatus.textContent = isVideo
            ? 'Decoding video stream & running YOLOv8 tracking...'
            : 'Running vehicle detection & plate localization...';
          if (processingInnerIcon) processingInnerIcon.textContent = '🔍';
        } else if (elapsedSec < 13) {
          processingStatus.textContent = 'Reading number plates with EasyOCR + TrOCR brain...';
          if (processingInnerIcon) processingInnerIcon.textContent = '🧠';
        } else {
          processingStatus.textContent = 'Verifying state jurisdiction & consensus filtering...';
          if (processingInnerIcon) processingInnerIcon.textContent = '📋';
        }
      }
    }, 1000);
  }

  function hideProcessingOverlay(success = true) {
    if (processingInterval) {
      clearInterval(processingInterval);
      processingInterval = null;
    }
    if (!processingModal) return;

    if (success) {
      if (processingCircleSpinner) {
        processingCircleSpinner.style.stroke = 'var(--status-ok)';
      }
      if (processingInnerIcon) processingInnerIcon.textContent = '✅';
      if (processingStatus) {
        processingStatus.textContent = 'Analysis complete! Rendering results...';
        processingStatus.style.color = 'var(--status-ok)';
      }
      setTimeout(() => {
        processingModal.classList.remove('active');
        setTimeout(() => {
          processingModal.style.display = 'none';
        }, 250);
      }, 500);
    }
  }

  function showProcessingError(errorMessage) {
    if (processingInterval) {
      clearInterval(processingInterval);
      processingInterval = null;
    }
    if (!processingModal) return;

    if (processingCircleSpinner) {
      processingCircleSpinner.style.stroke = 'var(--status-danger)';
      processingCircleSpinner.style.animation = 'none';
      processingCircleSpinner.style.strokeDashoffset = '0';
    }
    if (processingInnerIcon) processingInnerIcon.textContent = '❌';
    if (processingTitle) {
      processingTitle.textContent = 'ANALYSIS ENCOUNTERED AN ERROR';
      processingTitle.style.color = 'var(--status-danger)';
    }
    if (processingStatus) {
      processingStatus.textContent = 'The model could not finish analyzing the file.';
      processingStatus.style.color = 'var(--status-danger)';
    }
    if (processingServerBadge) {
      processingServerBadge.textContent = 'ERROR STOP';
      processingServerBadge.style.color = 'var(--status-danger)';
      processingServerBadge.style.borderColor = 'var(--status-danger-border)';
      processingServerBadge.style.background = 'var(--status-danger-bg)';
    }
    if (processingTimerDot) {
      processingTimerDot.style.background = 'var(--status-danger)';
      processingTimerDot.style.boxShadow = '0 0 6px var(--status-danger)';
    }
    if (processingReassurance) processingReassurance.style.display = 'none';
    if (processingErrorBox) processingErrorBox.style.display = 'block';
    if (processingErrorMessage) {
      processingErrorMessage.textContent = errorMessage || 'Unknown network or server error.';
    }
  }

  if (btnDismissError) {
    btnDismissError.addEventListener('click', () => {
      if (processingModal) {
        processingModal.classList.remove('active');
        setTimeout(() => {
          processingModal.style.display = 'none';
        }, 250);
      }
    });
  }

  // -------------------------------------------------------------
  // Send File or Blob to Backend /analyze-parking
  // -------------------------------------------------------------
  async function sendInferenceRequest(fileOrBlob, filename, isVideo = false) {
    showProcessingOverlay(filename, isVideo);

    const formData = new FormData();
    formData.append('file', fileOrBlob, filename);

    const confSlider = document.getElementById('confSliderMobile');
    const confVal = confSlider ? parseFloat(confSlider.value) : 0.40;

    try {
      const res = await fetch(`${API_BASE}/analyze-parking?conf=${confVal}`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        let errDetails = `HTTP ${res.status}`;
        try {
          const errJson = await res.json();
          if (errJson && errJson.error) {
            errDetails = errJson.error;
          }
        } catch (_) {
          const errText = await res.text();
          if (errText) errDetails = errText;
        }
        throw new Error(errDetails);
      }

      const data = await res.json();
      if (data.status === 'error') {
        throw new Error(data.error || 'Server reported analysis error.');
      }

      hideProcessingOverlay(true);
      renderInferenceResults(data);
      return data;
    } catch (err) {
      console.error('Inference error:', err);
      showProcessingError(err.message);
      showToast(`Analysis failed: ${err.message}`, 5000);
    }
  }

  // -------------------------------------------------------------
  // Render Inference Results (Metrics, Alerts, Media, Ledger)
  // -------------------------------------------------------------
  function renderInferenceResults(data) {
    if (!data || data.status !== 'success') return;

    // 1. Update Metrics
    const m = data.metrics || {};
    if (mTotalVehicles) mTotalVehicles.textContent = m.total_vehicles ?? 0;
    if (mTotalCars) mTotalCars.textContent = (m.cars_and_trucks || 0) + (m.two_wheelers || 0);
    if (mTotalBuses) mTotalBuses.textContent = m.buses ?? 0;

    // 2. Update Directives / Alerts
    const alerts = data.actionable_alerts || {};
    if (mKitchenAlert && alerts.kitchen) {
      mKitchenAlert.textContent = alerts.kitchen;
      if (m.buses > 0) {
        mKitchenCard.classList.add('urgent');
      } else {
        mKitchenCard.classList.remove('urgent');
      }
    }
    if (mFloorAlert && alerts.floor_manager) {
      mFloorAlert.textContent = alerts.floor_manager;
    }

    // 3. Render Media Viewport
    if (data.annotated_media && resultMediaWrap) {
      resultCard.style.display = 'block';
      if (data.media_type === 'video') {
        resultMediaWrap.innerHTML = `
          <video controls autoplay playsinline loop style="width:100%; border-radius:8px;">
            <source src="data:video/mp4;base64,${data.annotated_media}" type="video/mp4">
          </video>`;
      } else {
        resultMediaWrap.innerHTML = `
          <img src="data:image/jpeg;base64,${data.annotated_media}" alt="Annotated Capture" style="width:100%; border-radius:8px;">`;
      }
      if (resultInfoTag) {
        resultInfoTag.textContent = `${data.execution_time_ms}ms · ${data.number_plates.length} Plates`;
      }
    }

    // 4. Update Ledger
    if (Array.isArray(data.number_plates)) {
      data.number_plates.forEach(p => {
        // Prevent immediate duplicates in local ledger
        const exists = allLedgerPlates.some(item => item.text === p.text);
        if (!exists) {
          allLedgerPlates.unshift({
            ...p,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          });
        }
      });
      renderLedgerCards();
    }

    showToast(`Done: ${data.metrics.total_vehicles} Vehicles, ${data.number_plates.length} Plates`);
  }

  // -------------------------------------------------------------
  // Render Plate Ledger Cards (Strict Contrast Rules for Dark & Light)
  // -------------------------------------------------------------
  function renderLedgerCards() {
    if (!mobilePlateList) return;
    const query = (searchPlatesMobile ? searchPlatesMobile.value : '').trim().toLowerCase();

    const filtered = allLedgerPlates.filter(item => {
      if (!query) return true;
      const textMatch = item.text.toLowerCase().includes(query);
      const stateMatch = item.state_name && item.state_name.toLowerCase().includes(query);
      const codeMatch = item.state_code && item.state_code.toLowerCase().includes(query);
      return textMatch || stateMatch || codeMatch;
    });

    if (filtered.length === 0) {
      mobilePlateList.innerHTML = `
        <div style="text-align:center; padding: 24px; color: var(--text-muted); font-size:12px;">
          No license plates recorded yet. Capture or scan vehicles to populate the ledger.
        </div>`;
      return;
    }

    mobilePlateList.innerHTML = filtered.map(item => {
      const stateCode = item.state_code || 'IN';
      const stateName = item.state_name || 'Standard Registration';
      const conf = item.confidence_score ? `${item.confidence_score}%` : '98%';

      return `
        <div class="plate-item-mobile">
          <div class="plate-num-box">${item.text}</div>
          <div class="plate-jurisdiction">
            <span class="state-chip-mobile">📍 ${stateCode}</span>
            <span class="state-name-mobile">${stateName}</span>
          </div>
          <div class="plate-meta-mobile">
            <div class="conf-tag">${conf}</div>
            <div>${item.timestamp}</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // -------------------------------------------------------------
  // CSV Export (Direct from Supabase Database)
  // -------------------------------------------------------------
  function exportCsv() {
    const dateSelect = document.getElementById('mobileDateSelect');
    const selectedDate = dateSelect ? dateSelect.value : 'today';
    const actualParam = selectedDate === 'today' ? new Date().toISOString().slice(0, 10) : selectedDate;
    
    // Direct download from persistent Supabase backend database (available even days later)
    const downloadUrl = `/api/ledger/export-csv?date=${encodeURIComponent(actualParam)}`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `mothers_hut_ledger_${actualParam}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast(`Exported ${actualParam} CSV from Supabase.`);
  }

  async function loadMobileLedgerDates() {
    const dateSelect = document.getElementById('mobileDateSelect');
    if (!dateSelect) return;
    try {
      const res = await fetch('/api/ledger/dates');
      const data = await res.json();
      if (data && data.dates && data.dates.length > 0) {
        const todayIso = new Date().toISOString().slice(0, 10);
        let opts = `<option value="today">Today (${todayIso})</option>`;
        opts += `<option value="all">All Dates</option>`;
        data.dates.forEach(d => {
          opts += `<option value="${d.date}">${d.date} (${d.plate_count})</option>`;
        });
        dateSelect.innerHTML = opts;
      }
    } catch (e) {
      console.warn("Could not load dates:", e);
    }
  }

  async function loadMobilePersistentLedger(dateVal = null) {
    try {
      const todayIso = new Date().toISOString().slice(0, 10);
      const queryDate = (!dateVal || dateVal === "today") ? todayIso : dateVal;
      const res = await fetch(`/api/ledger/entries?date=${encodeURIComponent(queryDate)}`);
      const data = await res.json();
      if (data && data.entries && data.entries.length > 0) {
        allLedgerPlates = data.entries.map(e => ({
          text: e.plate_text,
          state_code: e.state_code,
          state_name: e.state_name,
          confidence_score: e.confidence_score,
          timestamp: e.timestamp,
          chunk: ""
        }));
        renderLedger(allLedgerPlates);
      } else if (dateVal && dateVal !== "today") {
        allLedgerPlates = [];
        renderLedger([]);
      }
    } catch (e) {
      console.warn("Could not load mobile persistent ledger:", e);
    }
  }

  // -------------------------------------------------------------
  // Event Bindings
  // -------------------------------------------------------------
  function initEvents() {
    // Live camera toggles
    if (btnToggleLiveCam) {
      btnToggleLiveCam.addEventListener('click', () => {
        if (mediaStream) {
          stopLiveCamera();
        } else {
          startLiveCamera();
        }
      });
    }

    if (btnFlipCam) {
      btnFlipCam.addEventListener('click', flipCamera);
    }

    if (btnSnapLiveFrame) {
      btnSnapLiveFrame.addEventListener('click', async () => {
        const blob = await captureVideoFrameBlob();
        if (blob) {
          showToast('Analyzing frame...');
          await sendInferenceRequest(blob, 'camera_snap.jpg', false);
        } else {
          showToast('Camera feed unavailable.');
        }
      });
    }

    if (btnRecordClip) {
      btnRecordClip.addEventListener('click', () => recordLiveClip(6));
    }

    if (btnLiveScanToggle) {
      btnLiveScanToggle.addEventListener('click', toggleLiveScanner);
    }

    // Native Mobile Video Hardware Trigger
    if (mobileVideoRecordInput) {
      mobileVideoRecordInput.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) {
          const fname = file.name || 'mobile_camera_clip.mp4';
          showToast(`Uploading video footage (${Math.round(file.size / 1024)} KB)...`);
          e.target.value = '';
          await sendInferenceRequest(file, fname, true);
        }
      });
    }

    // Native Mobile Photo Hardware Trigger
    if (mobilePhotoInput) {
      mobilePhotoInput.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) {
          const fname = file.name || 'mobile_photo.jpg';
          showToast(`Analyzing camera snap (${Math.round(file.size / 1024)} KB)...`);
          e.target.value = '';
          await sendInferenceRequest(file, fname, false);
        }
      });
    }

    // Standard File Upload
    if (fileUploadInput) {
      fileUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) {
          const isVid = file.type.startsWith('video') || file.name.match(/\.(mp4|avi|mov|webm|mkv|3gp)$/i);
          const fname = file.name || (isVid ? 'uploaded_video.mp4' : 'uploaded_image.jpg');
          showToast(`Uploading ${fname}...`);
          e.target.value = '';
          await sendInferenceRequest(file, fname, !!isVid);
        }
      });
    }

    // Search
    if (searchPlatesMobile) {
      searchPlatesMobile.addEventListener('input', renderLedgerCards);
    }

    // Date selector for historical ledger
    const mobileDateSelect = document.getElementById('mobileDateSelect');
    if (mobileDateSelect) {
      mobileDateSelect.addEventListener('change', (e) => {
        loadMobilePersistentLedger(e.target.value);
      });
    }

    // CSV Export
    if (btnExportCsvMobile) {
      btnExportCsvMobile.addEventListener('click', exportCsv);
    }
  }

  // -------------------------------------------------------------
  // Initial Boot
  // -------------------------------------------------------------
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initEvents();
    loadMobileLedgerDates();
    loadMobilePersistentLedger();
    checkHealth();
    setInterval(checkHealth, 8000);
  });

})();

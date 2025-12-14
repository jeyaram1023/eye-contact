// ========================= DOM ELEMENTS =========================
const video = document.getElementById('video');
const canvasOutput = document.getElementById('canvasOutput');
const canvasContext = canvasOutput.getContext('2d');
const eyeStatusEl = document.getElementById('eyeStatus');
const btStatusEl = document.getElementById('btStatus');
const connectBtn = document.getElementById('connectBtn');

// ========================= OPENCV VARIABLES =====================
let faceCascade;
let eyeCascade;
let streaming = false;

// ========================= BLUETOOTH ============================
const SERVICE_UUID = 0xFFE0;
const CHARACTERISTIC_UUID = 0xFFE1;
let bleDevice = null;
let bleCharacteristic = null;
let lastSentDirection = '';

// ========================= CONSTANTS ============================
const BLINK_RATIO_THRESHOLD = 0.25;
const GAZE_DEBOUNCE_MS = 150;
const GAZE_THRESHOLD_LOW = 0.40;
const GAZE_THRESHOLD_HIGH = 0.60;
const DEBUG_MODE = true;

// Haar cascades (public URLs)
const FACE_CASCADE_URL =
  'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml';
const EYE_CASCADE_URL =
  'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml';

// ========================= UTIL FUNCTIONS =======================
async function createFileFromUrl(url, fileName) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = new Uint8Array(await response.arrayBuffer());
    cv.FS_createDataFile('/', fileName, data, true, false, false);
    console.log('Loaded', fileName);
    return true;
  } catch (e) {
    console.error('Cascade load failed', fileName, e);
    updateEyeStatus('Error loading ' + fileName);
    return false;
  }
}

function updateEyeStatus(t) {
  eyeStatusEl.textContent = t;
}

function updateBtStatus(t) {
  btStatusEl.textContent = t;
}

// ========================= OPENCV INIT ==========================
function onCvReady() {
  updateEyeStatus('Loading models...');
  Promise.all([
    createFileFromUrl(FACE_CASCADE_URL, 'face.xml'),
    createFileFromUrl(EYE_CASCADE_URL, 'eye.xml')
  ]).then(([fOk, eOk]) => {
    if (!fOk || !eOk) {
      updateEyeStatus('Failed to load cascades');
      return;
    }
    faceCascade = new cv.CascadeClassifier();
    eyeCascade = new cv.CascadeClassifier();
    if (!faceCascade.load('face.xml') || !eyeCascade.load('eye.xml')) {
      updateEyeStatus('Failed to init classifiers');
      return;
    }
    updateEyeStatus('Starting camera...');
    startCamera();
  });
}

// ========================= CAMERA ===============================
function startCamera() {
  if (streaming) return;
  navigator.mediaDevices
    .getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    })
    .then(stream => {
      video.srcObject = stream;
      return video.play();
    })
    .catch(err => {
      console.error('Camera error', err);
      updateEyeStatus('Camera denied: ' + err.message);
    });
}

video.addEventListener('canplay', () => {
  if (streaming) return;
  canvasOutput.width = video.videoWidth;
  canvasOutput.height = video.videoHeight;
  streaming = true;
  updateEyeStatus('Processing...');
  requestAnimationFrame(processVideo);
});

// ========================= MAIN LOOP ============================
function processVideo() {
  if (!streaming || !faceCascade || !eyeCascade) {
    requestAnimationFrame(processVideo);
    return;
  }

  try {
    canvasContext.drawImage(video, 0, 0, canvasOutput.width, canvasOutput.height);
    let frame = cv.imread(canvasOutput);
    let gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);

    let faces = new cv.RectVector();
    let eyes = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, new cv.Size(30, 30));

    let direction = 'NO FACE';

    if (faces.size() > 0) {
      const faceRect = faces.get(0);
      cv.rectangle(
        frame,
        new cv.Point(faceRect.x, faceRect.y),
        new cv.Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height),
        [0, 255, 221, 255],
        3
      );

      let faceROI = gray.roi(faceRect);
      eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0, new cv.Size(10, 10));

      if (eyes.size() >= 2) {
        let eyeRects = [];
        for (let i = 0; i < eyes.size(); i++) eyeRects.push(eyes.get(i));
        eyeRects.sort((a, b) => a.x - b.x);
        const leftEyeRect = eyeRects[0];
        const rightEyeRect = eyeRects[1];

        const leftAspectRatio = leftEyeRect.height / leftEyeRect.width;
        if (leftAspectRatio < BLINK_RATIO_THRESHOLD) {
          direction = 'BLINK';
        } else {
          direction = detectGaze(faceROI, leftEyeRect, frame, faceRect);
        }

        for (let i = 0; i < 2; i++) {
          let e = eyeRects[i];
          cv.rectangle(
            frame,
            new cv.Point(faceRect.x + e.x, faceRect.y + e.y),
            new cv.Point(faceRect.x + e.x + e.width, faceRect.y + e.y + e.height),
            [0, 255, 0, 255],
            2
          );
        }
      }
      faceROI.delete();
    }

    updateEyeStatus(direction);
    sendDirection(direction);

    cv.imshow(canvasOutput, frame);
    frame.delete();
    gray.delete();
    faces.delete();
    eyes.delete();
  } catch (e) {
    console.error('process error', e);
    updateEyeStatus('Error: ' + e.message);
  }

  requestAnimationFrame(processVideo);
}

// ========================= GAZE DETECTION =======================
function detectGaze(faceROI, eyeRect, frame, faceRect) {
  let eyeROI = faceROI.roi(eyeRect);
  let th = new cv.Mat();
  let proc = new cv.Mat();

  cv.equalizeHist(eyeROI, eyeROI);
  cv.threshold(eyeROI, th, 50, 255, cv.THRESH_BINARY_INV);

  let kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
  cv.morphologyEx(th, proc, cv.MORPH_CLOSE, kernel);
  cv.morphologyEx(proc, proc, cv.MORPH_OPEN, kernel);

  let m = cv.moments(proc, true);
  let direction = 'CENTER';

  if (m.m00 > 100) {
    let cx = m.m10 / m.m00;
    let cy = m.m01 / m.m00;
    let ratio = cx / eyeROI.cols;

    // Video is mirrored (rotateY(180deg)), so invert meaning
    if (ratio < GAZE_THRESHOLD_LOW) direction = 'LEFT';
    else if (ratio > GAZE_THRESHOLD_HIGH) direction = 'RIGHT';
    else direction = 'CENTER';

    if (DEBUG_MODE) {
      let px = faceRect.x + eyeRect.x + cx;
      let py = faceRect.y + eyeRect.y + cy;
      cv.circle(frame, new cv.Point(px, py), 3, [0, 255, 0, 255], -1);
      cv.putText(
        frame,
        'Ratio: ' + ratio.toFixed(2),
        new cv.Point(20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        [0, 255, 221, 255],
        2
      );
    }
  }

  kernel.delete();
  th.delete();
  proc.delete();
  eyeROI.delete();
  return direction;
}

// ========================= BLUETOOTH ============================
connectBtn.addEventListener('click', async () => {
  if (!bleDevice || !bleDevice.gatt.connected) await connectBluetooth();
  else await disconnectBluetooth();
});

async function connectBluetooth() {
  try {
    updateBtStatus('Requesting device...');
    bleDevice = await navigator.bluetooth.requestDevice({
      filters: [{ services: [SERVICE_UUID] }],
      optionalServices: [SERVICE_UUID]
    });
    updateBtStatus('Connecting...');
    const server = await bleDevice.gatt.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    bleCharacteristic = await service.getCharacteristic(CHARACTERISTIC_UUID);
    updateBtStatus('Connected: ' + bleDevice.name);
    connectBtn.textContent = '🔌 Disconnect';
    bleDevice.addEventListener('gattserverdisconnected', onDisconnected);
  } catch (e) {
    console.error('BT error', e);
    updateBtStatus('Error: ' + e.message);
    if (bleDevice && bleDevice.gatt.connected) bleDevice.gatt.disconnect();
  }
}

async function disconnectBluetooth() {
  if (bleDevice && bleDevice.gatt.connected) bleDevice.gatt.disconnect();
}

function onDisconnected() {
  updateBtStatus('Disconnected');
  connectBtn.textContent = '🔵 Connect Bluetooth';
  bleDevice = null;
  bleCharacteristic = null;
  lastSentDirection = '';
}

function sendDirection(direction) {
  if (!bleCharacteristic) return;
  if (direction === 'NO FACE') return;
  if (direction === lastSentDirection) return;

  lastSentDirection = direction;
  setTimeout(() => {
    if (lastSentDirection === direction) {
      try {
        const enc = new TextEncoder();
        bleCharacteristic.writeValue(enc.encode(direction));
        console.log('Sent:', direction);
      } catch (e) {
        console.error('Send error', e);
      }
    }
  }, GAZE_DEBOUNCE_MS);
}

// ========================= OPENCV HOOK ==========================
window.Module = {
  onRuntimeInitialized() {
    onCvReady();
  }
};

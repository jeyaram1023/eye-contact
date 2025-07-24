// DOM Elements
const video = document.getElementById('video');
const canvasOutput = document.getElementById('canvasOutput');
const canvasContext = canvasOutput.getContext('2d');
const eyeStatusEl = document.getElementById('eyeStatus');
const btStatusEl = document.getElementById('btStatus');
const connectBtn = document.getElementById('connectBtn');

// OpenCV variables
let faceCascade;
let eyeCascade;
let streaming = false;

// Bluetooth variables
const SERVICE_UUID = 0xFFE0;
const CHARACTERISTIC_UUID = 0xFFE1;
let bleDevice = null;
let bleCharacteristic = null;
let lastSentDirection = '';

// Eye Tracking constants
const BLINK_RATIO_THRESHOLD = 0.35; // Aspect ratio (height/width) for blink detection
const GAZE_DEBOUNCE_MS = 100; // Debounce time for sending gaze commands

// --- UTILITY FUNCTIONS ---

/**
 * Helper function to load cascade files from URL.
 * OpenCV.js requires files to be in its own virtual file system.
 * @param {string} url - The URL of the cascade file.
 * @param {string} fileName - The name to save the file as in the virtual file system.
 */
async function createFileFromUrl(url, fileName) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.arrayBuffer();
        const dataArr = new Uint8Array(data);
        cv.FS_createDataFile('/', fileName, dataArr, true, false, false);
    } catch (error) {
        console.error(`Failed to load cascade file: ${fileName}`, error);
        updateEyeStatus(`Error loading ${fileName}`);
    }
}

/**
 * Updates the eye detection status on the UI.
 * @param {string} text - The status message to display.
 */
function updateEyeStatus(text) {
    eyeStatusEl.textContent = text;
}

/**
 * Updates the Bluetooth connection status on the UI.
 * @param {string} text - The status message to display.
 */
function updateBtStatus(text) {
    btStatusEl.textContent = text;
}

// --- MAIN OPENCV FUNCTION ---

function onCvReady() {
    updateEyeStatus("OpenCV.js is ready. Loading cascade files...");
    // Load cascade classifiers
    Promise.all([
        createFileFromUrl('haarcascade_frontalface_default.xml', 'face.xml'),
        createFileFromUrl('haarcascade_eye_tree_eyeglasses.xml', 'eye.xml')
    ]).then(() => {
        faceCascade = new cv.CascadeClassifier();
        eyeCascade = new cv.CascadeClassifier();
        if (!faceCascade.load('face.xml') || !eyeCascade.load('eye.xml')) {
            updateEyeStatus("Error loading cascade classifiers. Check file paths and network.");
            return;
        }
        updateEyeStatus("Cascade files loaded. Starting camera...");
        startCamera();
    });
}

// --- WEBCAM AND VIDEO PROCESSING ---

function startCamera() {
    if (streaming) return;
    navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
    }).then(stream => {
        video.srcObject = stream;
        video.play();
    }).catch(err => {
        console.error("Camera Error:", err);
        updateEyeStatus(`Camera access denied: ${err.message}`);
    });
}

video.addEventListener('canplay', () => {
    if (!streaming) {
        // Set canvas dimensions to match video
        canvasOutput.width = video.videoWidth;
        canvasOutput.height = video.videoHeight;
        streaming = true;
        // Start the video processing loop
        requestAnimationFrame(processVideo);
    }
});

function processVideo() {
    if (!streaming) return;

    // Start processing timer
    const startTime = performance.now();

    // Capture a frame from the video
    canvasContext.drawImage(video, 0, 0, canvasOutput.width, canvasOutput.height);
    let frame = cv.imread(canvasOutput);
    let gray = new cv.Mat();
    cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

    let faces = new cv.RectVector();
    let eyes = new cv.RectVector();
    
    // Detect faces
    faceCascade.detectMultiScale(gray, faces);
    
    let direction = "CENTER"; // Default direction

    if (faces.size() > 0) {
        const faceRect = faces.get(0);
        // Draw rectangle around the face
        let point1 = new cv.Point(faceRect.x, faceRect.y);
        let point2 = new cv.Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height);
        cv.rectangle(frame, point1, point2, [0, 255, 221, 255], 3);

        // Get face Region of Interest (ROI) for eye detection
        let faceROI = gray.roi(faceRect);
        eyeCascade.detectMultiScale(faceROI, eyes);

        if (eyes.size() >= 2) {
            // Sort eyes by x position to identify left and right
            let eyeRects = [];
            for (let i = 0; i < eyes.size(); i++) {
                eyeRects.push(eyes.get(i));
            }
            eyeRects.sort((a, b) => a.x - b.x);

            const leftEyeRect = eyeRects[0];
            const rightEyeRect = eyeRects[1];

            // --- Blink Detection ---
            const leftAspectRatio = leftEyeRect.height / leftEyeRect.width;
            if (leftAspectRatio < BLINK_RATIO_THRESHOLD) {
                direction = "BLINK";
            } else {
                // --- Gaze Detection (using the left eye) ---
                const eyeROI = faceROI.roi(leftEyeRect);
                let thresholdedEye = new cv.Mat();
                // Threshold to isolate pupil/iris
                cv.threshold(eyeROI, thresholdedEye, 60, 255, cv.THRESH_BINARY_INV);

                // Find moments to calculate the center of the iris
                let moments = cv.moments(thresholdedEye, true);
                let cx = moments.m10 / moments.m00;
                
                const eyeCenterX = eyeROI.cols / 2;
                const gazeRatio = cx / eyeROI.cols;

                if (gazeRatio < 0.4) {
                    direction = "RIGHT"; // Flipped due to mirror view
                } else if (gazeRatio > 0.6) {
                    direction = "LEFT"; // Flipped due to mirror view
                } else {
                    direction = "CENTER";
                }
                
                // Cleanup eye mats
                eyeROI.delete();
                thresholdedEye.delete();
            }

            // Draw rectangles around eyes
            for(let i = 0; i < 2; i++) {
                let eye = eyeRects[i];
                let point1 = new cv.Point(faceRect.x + eye.x, faceRect.y + eye.y);
                let point2 = new cv.Point(faceRect.x + eye.x + eye.width, faceRect.y + eye.y + eye.height);
                cv.rectangle(frame, point1, point2, [0, 255, 0, 255], 2);
            }
        }
        faceROI.delete();
    } else {
      direction = "NO FACE";
    }

    updateEyeStatus(direction);
    sendDirection(direction);

    // Display the processed frame
    cv.imshow(canvasOutput, frame);
    
    // Cleanup main mats
    frame.delete();
    gray.delete();
    faces.delete();
    eyes.delete();

    // Call the next frame
    requestAnimationFrame(processVideo);
}


// --- BLUETOOTH FUNCTIONS ---

connectBtn.addEventListener('click', async () => {
    if (!bleDevice || !bleDevice.gatt.connected) {
        await connectBluetooth();
    } else {
        await disconnectBluetooth();
    }
});

async function connectBluetooth() {
    try {
        updateBtStatus("Requesting device...");
        bleDevice = await navigator.bluetooth.requestDevice({
            filters: [{ services: [SERVICE_UUID] }],
            optionalServices: [SERVICE_UUID]
        });

        updateBtStatus("Connecting to GATT Server...");
        const server = await bleDevice.gatt.connect();

        updateBtStatus("Getting Service...");
        const service = await server.getPrimaryService(SERVICE_UUID);

        updateBtStatus("Getting Characteristic...");
        bleCharacteristic = await service.getCharacteristic(CHARACTERISTIC_UUID);

        updateBtStatus(`Connected to ${bleDevice.name}`);
        connectBtn.textContent = '🔌 Disconnect';
        bleDevice.addEventListener('gattserverdisconnected', onDisconnected);

    } catch (error) {
        console.error("Bluetooth Error:", error);
        updateBtStatus(`Error: ${error.message}`);
        if(bleDevice) bleDevice.gatt.disconnect();
    }
}

async function disconnectBluetooth() {
    if (bleDevice && bleDevice.gatt.connected) {
        bleDevice.gatt.disconnect();
    }
}

function onDisconnected() {
    updateBtStatus("Disconnected");
    connectBtn.textContent = '🔵 Connect Bluetooth';
    bleDevice = null;
    bleCharacteristic = null;
    lastSentDirection = '';
}

function sendDirection(direction) {
    if (bleCharacteristic && direction !== lastSentDirection && direction !== "NO FACE") {
        lastSentDirection = direction;
        setTimeout(() => {
            // Only send if the direction is still the same after a short delay
            if (lastSentDirection === direction) {
                try {
                    const encoder = new TextEncoder();
                    bleCharacteristic.writeValue(encoder.encode(direction));
                    console.log(`Sent: ${direction}`);
                } catch (error) {
                    console.error("Failed to send data:", error);
                }
            }
        }, GAZE_DEBOUNCE_MS);
    }
}

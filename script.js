// ============================================
// DOM Elements
// ============================================
const video = document.getElementById('video');
const canvasOutput = document.getElementById('canvasOutput');
const canvasContext = canvasOutput.getContext('2d');
const eyeStatusEl = document.getElementById('eyeStatus');
const btStatusEl = document.getElementById('btStatus');
const connectBtn = document.getElementById('connectBtn');

// ============================================
// OpenCV Variables
// ============================================
let faceCascade;
let eyeCascade;
let streaming = false;

// ============================================
// Bluetooth Variables
// ============================================
const SERVICE_UUID = 0xFFE0;
const CHARACTERISTIC_UUID = 0xFFE1;
let bleDevice = null;
let bleCharacteristic = null;
let lastSentDirection = '';

// ============================================
// Eye Tracking Constants
// ============================================
const BLINK_RATIO_THRESHOLD = 0.35;
const GAZE_DEBOUNCE_MS = 100;
const GAZE_THRESHOLD_LOW = 0.35;   // Lower threshold for LEFT detection
const GAZE_THRESHOLD_HIGH = 0.65;  // Higher threshold for RIGHT detection

// ============================================
// Cascade File URLs (GitHub CDN)
// ============================================
const FACE_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml';
const EYE_CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml';

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Load cascade XML files from URL into OpenCV virtual file system
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
        console.log(`✓ Successfully loaded: ${fileName}`);
        return true;
    } catch (error) {
        console.error(`✗ Failed to load cascade file: ${fileName}`, error);
        updateEyeStatus(`Error loading ${fileName}`);
        return false;
    }
}

/**
 * Update eye status display
 */
function updateEyeStatus(text) {
    eyeStatusEl.textContent = text;
    console.log('Eye Status:', text);
}

/**
 * Update Bluetooth status display
 */
function updateBtStatus(text) {
    btStatusEl.textContent = text;
    console.log('BT Status:', text);
}

// ============================================
// OPENCV INITIALIZATION
// ============================================

/**
 * Called when OpenCV.js runtime is ready
 */
function onCvReady() {
    updateEyeStatus('Loading cascade files...');
    console.log('OpenCV.js is ready');
    
    Promise.all([
        createFileFromUrl(FACE_CASCADE_URL, 'face.xml'),
        createFileFromUrl(EYE_CASCADE_URL, 'eye.xml')
    ]).then(([faceLoaded, eyeLoaded]) => {
        if (!faceLoaded || !eyeLoaded) {
            updateEyeStatus('Error: Cascade files failed to load');
            return;
        }

        // Load cascades into OpenCV
        faceCascade = new cv.CascadeClassifier();
        eyeCascade = new cv.CascadeClassifier();
        
        const faceLoaded_cv = faceCascade.load('face.xml');
        const eyeLoaded_cv = eyeCascade.load('eye.xml');
        
        if (!faceLoaded_cv || !eyeLoaded_cv) {
            updateEyeStatus('Error: Failed to load classifiers');
            console.error('Cascade load error. Face:', faceLoaded_cv, 'Eye:', eyeLoaded_cv);
            return;
        }
        
        updateEyeStatus('Ready - Requesting camera...');
        startCamera();
    }).catch(error => {
        console.error('Cascade loading error:', error);
        updateEyeStatus('Error: ' + error.message);
    });
}

// ============================================
// CAMERA INITIALIZATION
// ============================================

/**
 * Request camera access and start video stream
 */
function startCamera() {
    if (streaming) return;
    
    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: { ideal: 'user' }
        },
        audio: false
    };
    
    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            video.srcObject = stream;
            video.play();
            console.log('✓ Camera started');
        })
        .catch(err => {
            console.error('Camera Error:', err);
            updateEyeStatus(`Camera denied: ${err.message}`);
        });
}

/**
 * When video is ready, start processing
 */
video.addEventListener('canplay', () => {
    if (!streaming) {
        canvasOutput.width = video.videoWidth;
        canvasOutput.height = video.videoHeight;
        streaming = true;
        console.log('✓ Video is ready. Canvas:', canvasOutput.width, 'x', canvasOutput.height);
        updateEyeStatus('Processing...');
        requestAnimationFrame(processVideo);
    }
});

// ============================================
// VIDEO PROCESSING - MAIN LOOP
// ============================================

/**
 * Main video processing loop
 * - Capture frame
 * - Detect faces
 * - Detect eyes
 * - Determine gaze direction
 * - Send data via Bluetooth
 */
function processVideo() {
    if (!streaming || !faceCascade || !eyeCascade) {
        requestAnimationFrame(processVideo);
        return;
    }

    try {
        // Draw video frame to canvas
        canvasContext.drawImage(video, 0, 0, canvasOutput.width, canvasOutput.height);
        
        // Read frame as OpenCV Mat
        let frame = cv.imread(canvasOutput);
        let gray = new cv.Mat();
        cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);

        let faces = new cv.RectVector();
        let eyes = new cv.RectVector();
        
        // Detect faces
        faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0, new cv.Size(30, 30));
        
        let direction = 'NO FACE';

        if (faces.size() > 0) {
            const faceRect = faces.get(0);
            
            // Draw face rectangle
            let point1 = new cv.Point(faceRect.x, faceRect.y);
            let point2 = new cv.Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height);
            cv.rectangle(frame, point1, point2, [0, 255, 221, 255], 3);

            // Get face ROI for eye detection
            let faceROI = gray.roi(faceRect);
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 4, 0, new cv.Size(15, 15));

            if (eyes.size() >= 2) {
                // Sort eyes by x position (left to right)
                let eyeRects = [];
                for (let i = 0; i < eyes.size(); i++) {
                    eyeRects.push(eyes.get(i));
                }
                eyeRects.sort((a, b) => a.x - b.x);

                const leftEyeRect = eyeRects[0];
                const rightEyeRect = eyeRects[1];

                // --- BLINK DETECTION ---
                const leftAspectRatio = leftEyeRect.height / leftEyeRect.width;
                
                if (leftAspectRatio < BLINK_RATIO_THRESHOLD) {
                    direction = 'BLINK';
                } else {
                    // --- GAZE DETECTION ---
                    const eyeROI = faceROI.roi(leftEyeRect);
                    let thresholdedEye = new cv.Mat();
                    
                    // Threshold to isolate pupil/iris
                    cv.threshold(eyeROI, thresholdedEye, 60, 255, cv.THRESH_BINARY_INV);

                    // Calculate iris center using moments
                    let moments = cv.moments(thresholdedEye, true);
                    
                    if (moments.m00 > 0) {
                        let cx = moments.m10 / moments.m00;
                        const gazeRatio = cx / eyeROI.cols;

                        // VIDEO IS FLIPPED (transform: rotateY(180deg))
                        // So LEFT/RIGHT are inverted in pixel space
                        if (gazeRatio < GAZE_THRESHOLD_LOW) {
                            direction = 'LEFT';   // Inverted because video is mirrored
                        } else if (gazeRatio > GAZE_THRESHOLD_HIGH) {
                            direction = 'RIGHT';  // Inverted because video is mirrored
                        } else {
                            direction = 'CENTER';
                        }
                    } else {
                        direction = 'CENTER';
                    }
                    
                    eyeROI.delete();
                    thresholdedEye.delete();
                }

                // Draw rectangles around both eyes
                for (let i = 0; i < 2; i++) {
                    let eye = eyeRects[i];
                    let point1 = new cv.Point(faceRect.x + eye.x, faceRect.y + eye.y);
                    let point2 = new cv.Point(faceRect.x + eye.x + eye.width, faceRect.y + eye.y + eye.height);
                    cv.rectangle(frame, point1, point2, [0, 255, 0, 255], 2);
                }
            }
            faceROI.delete();
        }

        // Update UI and send direction
        updateEyeStatus(direction);
        sendDirection(direction);

        // Display processed frame
        cv.imshow(canvasOutput, frame);
        
        // Cleanup
        frame.delete();
        gray.delete();
        faces.delete();
        eyes.delete();

    } catch (error) {
        console.error('Processing error:', error);
        updateEyeStatus('Error: ' + error.message);
    }

    requestAnimationFrame(processVideo);
}

// ============================================
// BLUETOOTH FUNCTIONS
// ============================================

/**
 * Connect/Disconnect button click handler
 */
connectBtn.addEventListener('click', async () => {
    if (!bleDevice || !bleDevice.gatt.connected) {
        await connectBluetooth();
    } else {
        await disconnectBluetooth();
    }
});

/**
 * Establish Bluetooth connection
 */
async function connectBluetooth() {
    try {
        updateBtStatus('Requesting device...');
        bleDevice = await navigator.bluetooth.requestDevice({
            filters: [{ services: [SERVICE_UUID] }],
            optionalServices: [SERVICE_UUID]
        });

        updateBtStatus('Connecting...');
        const server = await bleDevice.gatt.connect();

        updateBtStatus('Getting service...');
        const service = await server.getPrimaryService(SERVICE_UUID);

        updateBtStatus('Getting characteristic...');
        bleCharacteristic = await service.getCharacteristic(CHARACTERISTIC_UUID);

        updateBtStatus(`Connected: ${bleDevice.name}`);
        connectBtn.textContent = '🔌 Disconnect';
        bleDevice.addEventListener('gattserverdisconnected', onDisconnected);

    } catch (error) {
        console.error('Bluetooth Error:', error);
        updateBtStatus(`Error: ${error.message}`);
        if (bleDevice) bleDevice.gatt.disconnect();
    }
}

/**
 * Disconnect Bluetooth
 */
async function disconnectBluetooth() {
    if (bleDevice && bleDevice.gatt.connected) {
        bleDevice.gatt.disconnect();
    }
}

/**
 * Handle disconnection event
 */
function onDisconnected() {
    updateBtStatus('Disconnected');
    connectBtn.textContent = '🔵 Connect Bluetooth';
    bleDevice = null;
    bleCharacteristic = null;
    lastSentDirection = '';
}

/**
 * Send eye direction to Bluetooth device
 */
function sendDirection(direction) {
    if (bleCharacteristic && direction !== lastSentDirection && direction !== 'NO FACE') {
        lastSentDirection = direction;
        setTimeout(() => {
            if (lastSentDirection === direction) {
                try {
                    const encoder = new TextEncoder();
                    bleCharacteristic.writeValue(encoder.encode(direction));
                    console.log(`📤 Sent: ${direction}`);
                } catch (error) {
                    console.error('Failed to send:', error);
                }
            }
        }, GAZE_DEBOUNCE_MS);
    }
}

// ============================================
// OPENCV RUNTIME INITIALIZATION HOOK
// ============================================

/**
 * Called automatically when opencv.js loads
 */
window.Module = {
    onRuntimeInitialized() {
        console.log('✓ OpenCV.js runtime initialized');
        onCvReady();
    }
};

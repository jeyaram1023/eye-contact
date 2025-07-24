let video = document.getElementById("video");
let canvas = document.getElementById("canvasOutput");
let statusEl = document.getElementById("status");
let ctx = canvas.getContext("2d");

let streaming = false;
let src, gray, faceROI, eyeROI;
let faceCascade, eyeCascade;

function processVideo() {
  if (!streaming) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  src.data.set(ctx.getImageData(0, 0, canvas.width, canvas.height).data);

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  let faces = new cv.RectVector();
  faceCascade.detectMultiScale(gray, faces, 1.1, 3);

  for (let i = 0; i < faces.size(); ++i) {
    let face = faces.get(i);
    let faceMat = gray.roi(face);

    let eyes = new cv.RectVector();
    eyeCascade.detectMultiScale(faceMat, eyes);

    for (let j = 0; j < eyes.size(); ++j) {
      let eye = eyes.get(j);
      let eyeRect = new cv.Rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
      cv.rectangle(src, eyeRect, [255, 0, 0, 255], 2);

      // Get eye ROI
      let eyeMat = gray.roi(eyeRect);
      let eyeMean = cv.mean(eyeMat);
      let brightness = eyeMean[0];

      if (brightness < 60) {
        statusEl.textContent = "Blink Detected";
      } else {
        // Direction (very basic based on center of intensity)
        let moments = cv.moments(eyeMat);
        if (moments.m00 !== 0) {
          let cx = moments.m10 / moments.m00;
          let cy = moments.m01 / moments.m00;

          if (cx < eye.width * 0.4) statusEl.textContent = "Looking Left";
          else if (cx > eye.width * 0.6) statusEl.textContent = "Looking Right";
          else if (cy < eye.height * 0.4) statusEl.textContent = "Looking Up";
          else if (cy > eye.height * 0.6) statusEl.textContent = "Looking Down";
          else statusEl.textContent = "Looking Center";
        }
      }

      eyeMat.delete();
    }

    faceMat.delete();
    eyes.delete();
  }

  cv.imshow("canvasOutput", src);
  faces.delete();
  requestAnimationFrame(processVideo);
}

function startCamera() {
  navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
    video.srcObject = stream;
    video.play();
  });
}

video.addEventListener("play", () => {
  streaming = true;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  src = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
  gray = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC1);

  faceCascade = new cv.CascadeClassifier();
  eyeCascade = new cv.CascadeClassifier();

  // Load Haar cascades
  let faceCascadeFile = "haarcascade_frontalface_default.xml";
  let eyeCascadeFile = "haarcascade_eye_tree_eyeglasses.xml";

  cv.FS_createPreloadedFile("/", faceCascadeFile, faceCascadeFile, true, false);
  cv.FS_createPreloadedFile("/", eyeCascadeFile, eyeCascadeFile, true, false);

  faceCascade.load(faceCascadeFile);
  eyeCascade.load(eyeCascadeFile);

  requestAnimationFrame(processVideo);
});

startCamera();

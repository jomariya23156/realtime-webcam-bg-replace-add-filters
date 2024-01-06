const IMAGE_INTERVAL_MS = 42;
let uniqToken;

// =========== Utility functions ===========
// Function to fetch an image from the server and convert it to data URI
async function fetchAndConvertToDataURI(imagePath) {
  try {
    const response = await fetch(imagePath);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch image: ${response.status} ${response.statusText}`
      );
    }

    const blob = await response.blob();
    const base64Data = await blobToBase64(blob);
    return `data:${response.headers.get("content-type")};base64,${base64Data}`;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

async function readImageFromServer(imagePath) {
  try {
    const response = await fetch(imagePath);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch image: ${response.status} ${response.statusText}`
      );
    }

    const blob = await response.blob();
    return blob;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}

// Function to convert base64 to Blob
function base64ToBlob(base64Data, contentType) {
  const byteCharacters = atob(base64Data);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: contentType });
}
// =========================================

// =========== WebSocket related ===========
const startImageProcessing = (video, canvas, image, deviceId, sessionToken) => {
  const socket = new WebSocket(
    `ws://${location.host}/image_processing/${sessionToken}`
  );
  let intervalId;

  // Connection opened
  socket.addEventListener("open", function () {
    // Start reading video from device
    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          deviceId,
          width: { max: 640 },
          height: { max: 480 },
        },
      })
      .then(function (stream) {
        video.srcObject = stream;
        video.play().then(() => {
          // Adapt overlay canvas size to the video size
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          // Send an image in the WebSocket every 42 ms
          intervalId = setInterval(() => {
            // Create a virtual canvas to draw current video image
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // apply image filters
            ctx.filter = video.style.filter;
            ctx.drawImage(video, 0, 0);

            // convert it to JPEG and send it to the websocket
            canvas.toBlob((blob) => socket.send(blob), "image/jpeg");
          }, IMAGE_INTERVAL_MS);
        });
      });
  });

  // Listen for messages
  socket.addEventListener("message", function (event) {
    image.setAttribute("src", event.data);
  });

  socket.addEventListener("close", function () {
    window.clearInterval(intervalId);
    video.pause();
  });

  return socket;
};

window.addEventListener("DOMContentLoaded", (event) => {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const image = document.getElementById("processed-image");
  const cameraSelect = document.getElementById("camera-select");
  let socket;
  // generate a unique token here to use for this session
  // For this such simple case, current date works just fine
  uniqToken = Date.now();
  console.log("Token for this session:", uniqToken);

  // List available cameras and fill select
  navigator.mediaDevices.getUserMedia({ audio: true, video: true }).then(() => {
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      for (const device of devices) {
        if (device.kind === "videoinput" && device.deviceId) {
          const deviceOption = document.createElement("option");
          deviceOption.value = device.deviceId;
          deviceOption.innerText = device.label;
          cameraSelect.appendChild(deviceOption);
        }
      }
    });
  });

  document
    .getElementById("form-connect")
    .addEventListener("submit", (event) => {
      event.preventDefault();

      // Close previous socket if there is one
      if (socket) {
        socket.close();
      }

      const deviceId = cameraSelect.selectedOptions[0].value;
      socket = startImageProcessing(video, canvas, image, deviceId, uniqToken);
    });
});
// =========================================

// ============= REST related =============
const radioOptions = document.querySelectorAll(
  'input[name="cartoonifier-option"]'
);
radioOptions.forEach((radio) =>
  radio.addEventListener("change", submitCartoonifierOption)
);

function submitCartoonifierOption(event) {
  const selectedOption = event.target.value;
  const apiUrl = `http://${location.host}/change_cartoonify/${selectedOption}`;
  const headers = new Headers();
  headers.append("X-Session-Token", uniqToken);

  fetch(apiUrl, {
    method: "PUT",
    headers: headers,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Cartoonifier Option Response:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

async function submitImageBg() {
  // Gather necessary data (e.g., selected image or uploaded image data)
  const selectedImage = document.querySelector(".selected-image img");
  const uploadedImage = document.querySelector("#preview-container img");

  let imageData;
  let blob;
  if (selectedImage) {
    console.log("Request with the prepared image.");
    imageData = selectedImage.src;
    blob = await readImageFromServer(imageData);
  } else if (uploadedImage) {
    console.log("Request with the uploaded image.");
    imageData = uploadedImage.src;
    const base64Data = imageData.split(",")[1]; // Remove the data URI prefix
    blob = base64ToBlob(base64Data, "image/jpeg"); // Convert base64 to Blob
  } else {
    console.error("No image selected or uploaded.");
    return;
  }

  // Make a POST request to the API endpoint
  const apiUrl = `http://${location.host}/change_bg`;
  const formData = new FormData();
  formData.append("file", blob, "image.jpg");
  const headers = new Headers();
  headers.append("X-Session-Token", uniqToken);

  fetch(apiUrl, {
    method: "POST",
    headers: headers,
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("API Response:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function requestResetBg(event) {
  document
    .querySelectorAll(".image")
    .forEach((img) => img.classList.remove("selected-image"));
  const previewContainer = document.getElementById("preview-container");
  const allChilds = previewContainer.getElementsByTagName("*");
  for (let child of allChilds) {
    previewContainer.removeChild(child);
  }

  const apiUrl = `http://${location.host}/reset_bg`;
  const headers = new Headers();
  headers.append("X-Session-Token", uniqToken);

  fetch(apiUrl, {
    method: "POST",
    headers: headers,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Reset BG Response:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
// ========================================

const IMAGE_INTERVAL_MS = 42;

const startObjectDetection = (video, canvas, image, deviceId) => {
    const socket = new WebSocket(`ws://${location.host}/object-detection`);
    let intervalId;

    // Connection opened
    socket.addEventListener('open', function() {
        // Start reading video from device
        navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                deviceId,
                width: {max: 640},
                height: {max: 480}
            },
        }).then(function(stream){
            video.srcObject = stream;
            video.play().then(() => {
                // Adapt overlay canvas size to the video size
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Send an image in the WebSocket every 42 ms
                intervalId = setInterval(() => {

                    // Create a virtual canvas to draw current video image
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.drawImage(video, 0, 0);

                    // convert it to JPEG and send it to the websocket
                    canvas.toBlob((blob) => socket.send(blob), 'image/jpeg');
                }, IMAGE_INTERVAL_MS);
            });
        });
    });

    // Listen for messages
    socket.addEventListener('message', function(event){
        // drawObjects(video, canvas, JSON.parse(event.data));
        image.setAttribute('src', event.data);
    });

    socket.addEventListener('close', function(){
        window.clearInterval(intervalId);
        video.pause();
    });

    return socket;
};

window.addEventListener('DOMContentLoaded', (event) => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const image = document.getElementById('processed-image');
    const cameraSelect = document.getElementById('camera-select');
    let socket;

    // List available cameras and fill select
    navigator.mediaDevices.getUserMedia({audio: true, video: true}).then(() => {
        navigator.mediaDevices.enumerateDevices().then((devices) => {
            for (const device of devices){
                if (device.kind === 'videoinput' && device.deviceId){
                    const deviceOption = document.createElement('option');
                    deviceOption.value = device.deviceId;
                    deviceOption.innerText = device.label;
                    cameraSelect.appendChild(deviceOption);
                }
            }
        });
    });

    document.getElementById('form-connect').addEventListener('submit', (event) => {
        event.preventDefault();

        // Close previous socket if there is one
        if (socket){
            socket.close();
        }

        const deviceId = cameraSelect.selectedOptions[0].value;
        socket = startObjectDetection(video, canvas, image, deviceId);
    })
});
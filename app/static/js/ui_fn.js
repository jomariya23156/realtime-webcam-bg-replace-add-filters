// Function to fetch an image from the server and convert it to data URI
async function fetchAndConvertToDataURI(imagePath) {
    try {
        const response = await fetch(imagePath);
        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();
        const base64Data = await blobToBase64(blob);
        return `data:${response.headers.get('content-type')};base64,${base64Data}`;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

async function readImageFromServer(imagePath) {
    try {
        const response = await fetch(imagePath);
        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();
        return blob;
    } catch (error) {
        console.error('Error:', error);
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

function selectImage(imageUrl) {
    // Remove the border from all images
    document.querySelectorAll('.image').forEach(img => img.classList.remove('selected-image'));

    // Handle the selected image (e.g., display it or perform any other action)
    console.log('Selected Image: ' + imageUrl);
    displayPreview(imageUrl);

    // Add a border to the selected image
    const selectedId = imageUrl.substring(imageUrl.lastIndexOf('/')+1, imageUrl.lastIndexOf('.jpeg'))
    console.log('Selected image id:', selectedId)
    const selectedImage = document.getElementById(selectedId);
    if (selectedImage) {
        selectedImage.classList.add('selected-image');
    }
}

function uploadPreview(event) {
    // Remove all selected image
    document.querySelectorAll('.image').forEach(img => img.classList.remove('selected-image'));

    const fileInput = event.target;
    const file = fileInput.files[0];

    if (file) {
        // Perform any client-side processing with the uploaded file
        const reader = new FileReader();
        reader.onload = function (e) {
            const imageUrl = e.target.result;
            // Display the uploaded image or perform any other action
            console.log('Uploaded Image: ' + imageUrl);
            displayPreview(imageUrl);
        };

        reader.readAsDataURL(file);
    }
}

function displayPreview(imageUrl) {
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.createElement('img');
    previewImage.src = imageUrl;
    previewContainer.innerHTML = '';
    previewContainer.appendChild(previewImage);
}

async function submitRequest() {
    // Gather necessary data (e.g., selected image or uploaded image data)
    const selectedImage = document.querySelector('.selected-image img');
    const uploadedImage = document.querySelector('#preview-container img');

    let imageData;
    let blob;
    if (selectedImage) {
        console.log('Request with the prepared image.')
        imageData = selectedImage.src;
        console.log('imageData:', imageData)
        blob = await readImageFromServer(imageData);
    } else if (uploadedImage) {
        console.log('Request with the uploaded image.')
        imageData = uploadedImage.src;
        const base64Data = imageData.split(',')[1]; // Remove the data URI prefix
        blob = base64ToBlob(base64Data, 'image/jpeg'); // Convert base64 to Blob  
    } else {
        console.error('No image selected or uploaded.');
        return;
    }

    // Now, you can make a POST request to the dummy API endpoint
    const apiUrl = 'http://127.0.0.1:8000/change_bg';
    const formData = new FormData();
    console.log('typeof blob:',typeof(blob))
    formData.append('file', blob, 'image.jpg');

    fetch(apiUrl, {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log('API Response:', data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

const radioOptions = document.querySelectorAll('input[name="cartoonifier-option"]');
radioOptions.forEach(radio => radio.addEventListener('change', submitCartoonifierOption));

function submitCartoonifierOption(event) {
    const selectedOption = event.target.value;
    const apiUrl = `http://localhost:8000/change_cartoonify/${selectedOption}`;

    fetch(apiUrl, {
    method: 'PUT',
    })
    .then(response => response.json())
    .then(data => {
    console.log('Cartoonifier Option Response:', data);
    })
    .catch(error => {
    console.error('Error:', error);
    });
}

// Listen for changes in the filter sliders and apply filters to the processed image
const filterSliders = document.querySelectorAll('#image-filters input');
filterSliders.forEach(slider => slider.addEventListener('input', applyImageFilters));

function applyImageFilters() {
  const brightnessValue = document.getElementById('brightness').value;
  const contrastValue = document.getElementById('contrast').value;
  const grayscaleValue = document.getElementById('grayscale').value;
  const saturateValue = document.getElementById('saturate').value;

  const filters = `brightness(${brightnessValue}%) contrast(${contrastValue}%) grayscale(${grayscaleValue}%) saturate(${saturateValue}%)`;

  // Apply the filters to the processed image
  const processedImage = document.getElementById('processed-image');
  processedImage.style.filter = filters;
}

function resetFilters() {
    // Set default values for each filter
    document.getElementById('brightness').value = 100;
    document.getElementById('contrast').value = 100;
    document.getElementById('grayscale').value = 0;
    document.getElementById('saturate').value = 100;

    // Apply the default filters
    applyImageFilters();
  }
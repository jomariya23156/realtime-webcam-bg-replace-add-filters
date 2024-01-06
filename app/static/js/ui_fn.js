function selectImage(imageUrl) {
  // Remove the border from all images
  document
    .querySelectorAll(".image")
    .forEach((img) => img.classList.remove("selected-image"));

  // Handle the selected image (e.g., display it or perform any other action)
  console.log("Selected Image: " + imageUrl);
  displayPreview(imageUrl);

  // Add a border to the selected image
  const selectedId = imageUrl.substring(
    imageUrl.lastIndexOf("/") + 1,
    imageUrl.lastIndexOf(".jpeg")
  );
  console.log("Selected image id:", selectedId);
  const selectedImage = document.getElementById(selectedId);
  if (selectedImage) {
    selectedImage.classList.add("selected-image");
  }
}

function uploadPreview(event) {
  // Remove all selected image
  document
    .querySelectorAll(".image")
    .forEach((img) => img.classList.remove("selected-image"));

  const fileInput = event.target;
  const file = fileInput.files[0];

  if (file) {
    // Perform any client-side processing with the uploaded file
    const reader = new FileReader();
    reader.onload = function (e) {
      const imageUrl = e.target.result;
      // Display the uploaded image or perform any other action
      console.log("Uploaded Image type: " + imageUrl.split(',')[0]);
      displayPreview(imageUrl);
    };

    reader.readAsDataURL(file);
  }
}

function displayPreview(imageUrl) {
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.createElement("img");
  previewImage.src = imageUrl;
  previewContainer.innerHTML = "";
  previewContainer.appendChild(previewImage);
}

// Listen for changes in the filter sliders and apply filters to the processed image
const filterSliders = document.querySelectorAll("#image-filters input");
filterSliders.forEach((slider) =>
  slider.addEventListener("input", applyImageFilters)
);

function applyImageFilters() {
  const brightnessValue = document.getElementById("brightness").value;
  const contrastValue = document.getElementById("contrast").value;
  const grayscaleValue = document.getElementById("grayscale").value;
  const saturateValue = document.getElementById("saturate").value;

  const filters = `brightness(${brightnessValue}%) contrast(${contrastValue}%) grayscale(${grayscaleValue}%) saturate(${saturateValue}%)`;

  // Apply the filters to the input video
  const applyVideo = document.getElementById("video");
  applyVideo.style.filter = filters;
}

function resetFilters() {
  // Set default values for each filter
  document.getElementById("brightness").value = 100;
  document.getElementById("contrast").value = 100;
  document.getElementById("grayscale").value = 0;
  document.getElementById("saturate").value = 100;

  // Apply the default filters
  applyImageFilters();
}

document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const uploadBox = document.getElementById("uploadBox");
  const fileInput = document.getElementById("fileInput");
  const cameraBox = document.getElementById("cameraBox");
  const startCameraBtn = document.getElementById("startCamera");
  const cameraModal = document.getElementById("cameraModal");
  const closeModal = document.querySelectorAll(".close");
  const cameraFeed = document.getElementById("cameraFeed");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const captureBtn = document.getElementById("captureBtn");
  const retakeBtn = document.getElementById("retakeBtn");
  const useImageBtn = document.getElementById("useImageBtn");
  const resultsModal = document.getElementById("resultsModal");
  const resultImage = document.getElementById("resultImage");
  const detectionResults = document.getElementById("detectionResults");

  let stream = null;
  let capturedImage = null;

  // Event Listeners
  uploadBox.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", handleFileUpload);

  startCameraBtn.addEventListener("click", openCameraModal);

  closeModal.forEach((btn) => {
    btn.addEventListener("click", closeModalHandler);
  });

  captureBtn.addEventListener("click", captureImage);
  retakeBtn.addEventListener("click", retakeImage);
  useImageBtn.addEventListener("click", useCapturedImage);

  // Drag and drop functionality
  uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.classList.add("dragover");
  });

  uploadBox.addEventListener("dragleave", () => {
    uploadBox.classList.remove("dragover");
  });

  uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadBox.classList.remove("dragover");

    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileUpload();
    }
  });

  // Functions
  function handleFileUpload() {
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];

      // Validate file type
      const validTypes = ["image/jpeg", "image/jpg", "image/png"];
      if (!validTypes.includes(file.type)) {
        alert("Please upload a valid image file (JPEG, JPG, or PNG)");
        return;
      }

      // Process the image
      processImage(file);
    }
  }

  function openCameraModal() {
    cameraModal.style.display = "block";
    startCamera();
  }

  function closeModalHandler() {
    cameraModal.style.display = "none";
    resultsModal.style.display = "none";

    // Stop camera stream when modal is closed
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      cameraFeed.srcObject = stream;
    } catch (err) {
      console.error("Error accessing camera:", err);
      alert(
        "Could not access the camera. Please make sure you have granted camera permissions."
      );
      cameraModal.style.display = "none";
    }
  }

  function captureImage() {
    const context = cameraCanvas.getContext("2d");
    cameraCanvas.width = cameraFeed.videoWidth;
    cameraCanvas.height = cameraFeed.videoHeight;
    context.drawImage(
      cameraFeed,
      0,
      0,
      cameraCanvas.width,
      cameraCanvas.height
    );

    capturedImage = cameraCanvas.toDataURL("image/jpeg");

    // Show the captured image preview (you could show it in a preview element)
    captureBtn.style.display = "none";
    retakeBtn.style.display = "inline-block";
    useImageBtn.style.display = "inline-block";
  }

  function retakeImage() {
    captureBtn.style.display = "inline-block";
    retakeBtn.style.display = "none";
    useImageBtn.style.display = "none";
    capturedImage = null;
  }

  function useCapturedImage() {
    if (capturedImage) {
      // Convert data URL to blob
      fetch(capturedImage)
        .then((res) => res.blob())
        .then((blob) => {
          // Create a file from the blob
          const file = new File([blob], "captured.jpg", { type: "image/jpeg" });

          // Process the image
          processImage(file);

          // Close camera modal
          cameraModal.style.display = "none";
          if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
          }
        });
    }
  }

  function processImage(file) {
    // Show loading state
    detectionResults.innerHTML = "<p>Processing image... Please wait.</p>";
    resultsModal.style.display = "block";

    // Create FormData to send to Flask backend
    const formData = new FormData();
    formData.append("file", file);

    // Send image to Flask backend for processing
    fetch("/process_image", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          // Display the processed image
          resultImage.src = data.processed_image || URL.createObjectURL(file);

          // Display detection results
          if (data.detections && data.detections.length > 0) {
            let resultsHTML = "<h3>Detection Results:</h3><ul>";
            data.detections.forEach((detection) => {
              resultsHTML += `<li>${detection.class}: ${Math.round(
                detection.confidence * 100
              )}% confidence</li>`;
            });
            resultsHTML += "</ul>";
            detectionResults.innerHTML = resultsHTML;
          } else {
            detectionResults.innerHTML =
              "<p>No uniforms detected in the image.</p>";
          }
        } else {
          detectionResults.innerHTML = `<p>Error: ${
            data.error || "Unknown error occurred"
          }</p>`;
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        detectionResults.innerHTML =
          "<p>An error occurred while processing the image.</p>";
      });
  }

  // Close modal when clicking outside of it
  window.addEventListener("click", (event) => {
    if (event.target === cameraModal) {
      closeModalHandler();
    }
    if (event.target === resultsModal) {
      closeModalHandler();
    }
  });
});

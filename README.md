link of model : https://uniform-detection-model.streamlit.app/


# Uniform Detection Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

A web-based application for detecting college uniforms in images using a Convolutional Neural Network (CNN) model. The project includes a Flask backend for model inference and a user-friendly frontend with camera and image upload features.

---

## **Features**

- **Camera Input**: Capture images using your webcam and detect uniforms in real-time.
- **Image Upload**: Upload images from your computer for uniform detection.
- **Drag and Drop**: Drag and drop images for easy uploading.
- **Result Display**: Results are displayed in a modal dialog with a green checkmark (✔) for "Uniform Detected" or a red cross (❌) for "Non-Uniform Detected".
- **Back Button**: A back button allows users to re-upload images after viewing the result.

---

## **Technologies Used**

- **Backend**:
  - Python version 3.10.11
  - Flask~=3.1.0 (for web server)
  - tensorflow~=2.13.1 (for the CNN model)
  - opencv-python~=4.11.0.86 (for image processing)
  - matplotlib~=3.10.0
  - numpy~=1.24.3

- **Frontend**:
  - HTML, CSS, JavaScript
  - Font Awesome (for icons)

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rajveer-singh0/rajveer-singh0/uniform_detection.git
   cd uniform_detection

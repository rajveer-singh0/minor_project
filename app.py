# ................code 1................

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from werkzeug.utils import secure_filename
# import cv2

# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# # Load the trained binary classification model
# model = tf.keras.models.load_model('model/uniform_model.keras')
# CLASS_NAMES = ['non_uniform', 'uniform']  # 0 = non_uniform, 1 = uniform

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def predict_image(img_path):
#     try:
#         # Preprocess the image
#         img = image.load_img(img_path, target_size=(128, 128))
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Get prediction
#         prediction = model.predict(img_array)[0][0]  # Get scalar from [[0.73]]

#         # Determine label
#         label_idx = 1 if prediction > 0.5 else 0
#         label = CLASS_NAMES[label_idx]
#         confidence = prediction if label_idx == 1 else 1 - prediction

#         # Create a visualization of the result
#         img = cv2.imread(img_path)
#         if img is not None:
#             # Add text to the image
#             text = f"{label} ({confidence:.2%})"
#             cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                        0.7, (0, 255, 0), 2)
            
#             # Save the processed image
#             processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 
#                                       'processed_' + os.path.basename(img_path))
#             cv2.imwrite(processed_path, img)
#         else:
#             processed_path = img_path

#         return {
#             "label": label,
#             "confidence": round(confidence * 100, 2),
#             "processed_image": processed_path
#         }

#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return {
#             "label": "Error",
#             "confidence": 0,
#             "error": str(e)
#         }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file uploaded"})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"success": False, "error": "No selected file"})
    
#     if file and allowed_file(file.filename):
#         # Save the uploaded file
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Process the image
#             result = predict_image(filepath)
            
#             if result.get('error'):
#                 return jsonify({
#                     "success": False,
#                     "error": result['error']
#                 })
            
#             # Prepare response
#             response = {
#                 "success": True,
#                 "detections": [{
#                     "class": result['label'],
#                     "confidence": result['confidence'] / 100  # Convert to 0-1 range
#                 }],
#                 "processed_image": f"/{result['processed_image']}"
#             }
            
#             return jsonify(response)
            
#         except Exception as e:
#             return jsonify({"success": False, "error": str(e)})
    
#     return jsonify({"success": False, "error": "Invalid file type"})

# if __name__ == '__main__':
#     app.run(debug=True)



# ................code 2................

# import os

# # -----------------------
# # Environment configuration
# # -----------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (no CUDA)

# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from werkzeug.utils import secure_filename
# import cv2

# # -----------------------
# # Flask App Initialization
# # -----------------------
# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# # Load the trained binary classification model
# model = tf.keras.models.load_model('model/uniform_model.keras')
# CLASS_NAMES = ['non_uniform', 'uniform']  # 0 = non_uniform, 1 = uniform

# # -----------------------
# # Helper Functions
# # -----------------------
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def predict_image(img_path):
#     try:
#         # Preprocess the image
#         img = image.load_img(img_path, target_size=(128, 128))
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Prediction
#         prediction = model.predict(img_array)[0][0]
#         label_idx = 1 if prediction > 0.5 else 0
#         label = CLASS_NAMES[label_idx]
#         confidence = prediction if label_idx == 1 else 1 - prediction

#         # Visualization
#         img = cv2.imread(img_path)
#         if img is not None:
#             text = f"{label} ({confidence:.2%})"
#             cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (0, 255, 0), 2)
#             processed_path = os.path.join(
#                 app.config['UPLOAD_FOLDER'],
#                 'processed_' + os.path.basename(img_path)
#             )
#             cv2.imwrite(processed_path, img)
#         else:
#             processed_path = img_path

#         return {"label": label, "confidence": round(confidence * 100, 2), "processed_image": processed_path}

#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return {"label": "Error", "confidence": 0, "error": str(e)}

# # -----------------------
# # Routes
# # -----------------------
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/process_image', methods=['POST'])
# def process_image_route():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file uploaded"})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"success": False, "error": "No selected file"})

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         result = predict_image(filepath)
#         if result.get('error'):
#             return jsonify({"success": False, "error": result['error']})

#         response = {
#             "success": True,
#             "detections": [{
#                 "class": result['label'],
#                 "confidence": result['confidence'] / 100
#             }],
#             "processed_image": f"/{result['processed_image']}"
#         }
#         return jsonify(response)

#     return jsonify({"success": False, "error": "Invalid file type"})

# # -----------------------
# # Run App
# # -----------------------
# if __name__ == "__main__":
#     # Get port from environment variable or default to 5000
#     port = int(os.getenv("PORT", 5000))
#     # Debug mode off by default; can enable locally with FLASK_DEBUG=True
#     debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
#     app.run(host="0.0.0.0", port=port, debug=debug_mode)



# ........................Code 3......................


# import os
# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import cv2

# # -----------------------
# # Environment configuration
# # -----------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (no CUDA)

# # -----------------------
# # Flask App Initialization
# # -----------------------
# app = Flask(__name__)

# # Configuration
# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# # -----------------------
# # Load the trained model safely
# # -----------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, 'model', 'uniform_model.keras')

# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
# except Exception as e:
#     raise FileNotFoundError(
#         f"Model not found or incompatible. Make sure {MODEL_PATH} exists. Original error: {e}"
#     )

# CLASS_NAMES = ['non_uniform', 'uniform']  # 0 = non_uniform, 1 = uniform

# # -----------------------
# # Helper Functions
# # -----------------------
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def predict_image(img_path):
#     try:
#         # Preprocess the image
#         img = image.load_img(img_path, target_size=(128, 128))
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Prediction
#         prediction = model.predict(img_array)[0][0]
#         label_idx = 1 if prediction > 0.5 else 0
#         label = CLASS_NAMES[label_idx]
#         confidence = prediction if label_idx == 1 else 1 - prediction

#         # Visualization
#         img = cv2.imread(img_path)
#         if img is not None:
#             text = f"{label} ({confidence:.2%})"
#             cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (0, 255, 0), 2)
#             processed_path = os.path.join(
#                 app.config['UPLOAD_FOLDER'],
#                 'processed_' + os.path.basename(img_path)
#             )
#             cv2.imwrite(processed_path, img)
#         else:
#             processed_path = img_path

#         return {"label": label, "confidence": round(confidence * 100, 2), "processed_image": processed_path}

#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return {"label": "Error", "confidence": 0, "error": str(e)}

# # -----------------------
# # Routes
# # -----------------------
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/process_image', methods=['POST'])
# def process_image_route():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file uploaded"})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"success": False, "error": "No selected file"})

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         result = predict_image(filepath)
#         if result.get('error'):
#             return jsonify({"success": False, "error": result['error']})

#         response = {
#             "success": True,
#             "detections": [{
#                 "class": result['label'],
#                 "confidence": result['confidence'] / 100
#             }],
#             "processed_image": f"/{result['processed_image']}"
#         }
#         return jsonify(response)

#     return jsonify({"success": False, "error": "Invalid file type"})

# # -----------------------
# # Run App
# # -----------------------
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 5000))
#     debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
#     app.run(host="0.0.0.0", port=port, debug=debug_mode)





# code for streamlit 

import os
import numpy as np
import tensorflow as tf
import streamlit as st
import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# -----------------------
# Config
# -----------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

CLASS_NAMES = ['non_uniform', 'uniform']  # 0 = non_uniform, 1 = uniform
MODEL_PATH = os.path.join("model", "uniform_model.keras")

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Model not found or incompatible: {e}")
        st.stop()

model = load_model()

# -----------------------
# Prediction function
# -----------------------
def predict_image(file_bytes):
    try:
        # Load image from bytes
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        pil_img = pil_img.resize((128, 128))
        img_array = image.img_to_array(pil_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        label_idx = 1 if prediction > 0.5 else 0
        label = CLASS_NAMES[label_idx]
        confidence = prediction if label_idx == 1 else 1 - prediction

        # OpenCV visualization
        cv_img = np.array(pil_img)[:, :, ::-1].copy()  # convert RGB→BGR for cv2
        text = f"{label} ({confidence:.2%})"
        cv2.putText(cv_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "processed_image": cv_img[:, :, ::-1]  # back to RGB for Streamlit
        }

    except Exception as e:
        return {"label": "Error", "confidence": 0, "error": str(e)}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Uniform Detection", layout="centered")
st.title("👕 Uniform Detection System")
st.markdown("Upload an image to check if a person is wearing a uniform or not.")

uploaded_file = st.file_uploader("Upload an image (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    st.image(file_bytes, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect"):
        result = predict_image(file_bytes)

        if "error" in result:
            st.error(f"Prediction error: {result['error']}")
        else:
            st.success(f"Prediction: **{result['label']}** ({result['confidence']}% confidence)")
            st.image(result["processed_image"], caption="Processed Image with Prediction", use_column_width=True)
            st.json({
                "class": result["label"],
                "confidence": result["confidence"] / 100
            })

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# import sys
# import os

# IMG_HEIGHT, IMG_WIDTH = 128, 128
# CLASS_NAMES = ['non_uniform', 'uniform']  # 0: non_uniform, 1: uniform

# # Check command-line args
# if len(sys.argv) != 2:
#     print("Usage: python predict.py path_to_image")
#     sys.exit(1)

# img_path = sys.argv[1]

# # Check if file exists
# if not os.path.exists(img_path):
#     print(f"Image not found at {img_path}")
#     sys.exit(1)

# # Load model
# try:
#     model = tf.keras.models.load_model('model/uniform_model.keras')
# except Exception as e:
#     print(f"Error loading model: {e}")
#     sys.exit(1)

# # Load and preprocess image
# try:
#     img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
# except Exception as e:
#     print(f"Error processing image: {e}")
#     sys.exit(1)

# # Make prediction

# pred = model.predict(img_array)[0][0]  # Scalar between 0 and 1
# confidence = pred
# label = "uniform" if confidence > 0.4 else "non_uniform"

# print(f"Prediction: {label} (confidence: {confidence:.2f})")


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
MODEL_PATH = 'model/uniform_model.keras'
CLASS_NAMES = ['non_uniform', 'uniform']  # 0: non_uniform, 1: uniform

# Check usage
if len(sys.argv) != 2:
    print("Usage: python predict.py path_to_image")
    sys.exit(1)

img_path = sys.argv[1]

# Check if image exists
if not os.path.exists(img_path):
    print(f"❌ Image not found: {img_path}")
    sys.exit(1)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Load and preprocess image
try:
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
except Exception as e:
    print(f"❌ Error processing image: {e}")
    sys.exit(1)

# Make prediction
try:
    pred = model.predict(img_array)[0][0]  # scalar between 0 and 1
    label = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"✅ Prediction: {label} (confidence: {confidence * 100:.2f}%)")
except Exception as e:
    print(f"❌ Prediction error: {e}")
    sys.exit(1)

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Define custom metrics
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

# Register custom objects
with tf.keras.utils.custom_object_scope({'f1_m': f1_m, 'recall_m': recall_m, 'precision_m': precision_m}):
    model = tf.keras.models.load_model('my_model.h5')

# Categories for classification
CATEGORIES = ['Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J', 'Autorun.K',
              'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A', 'Fakerean', 'Gatak',
              'Instantaccess', 'Kelihos_ver1', 'Kelihos_ver3', 'Lollipop', 'Lolyda.AA1',
              'Lolyda.AA2', 'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.ACY',
              'Obfuscator.AD', 'Rammit', 'Rbot!gen', 'Simda', 'Skintrim.N', 'Swizzor.gen!E',
              'Swizzor.gen!I', 'Tracur', 'VB.AT', 'Vundo', 'Wintrim.BX', 'Yuner.A']

# Preprocessing function
def preprocess_image(image):
    # Convert image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
st.title("Malware Detection using Image")

# File uploader
uploaded_file = st.file_uploader("Upload an image for malware detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    
    # Get predicted category
    pred_name = CATEGORIES[np.argmax(prediction)]
    
    # Display the result
    st.write("Prediction:", pred_name)
    st.write("Prediction Confidence:", prediction)

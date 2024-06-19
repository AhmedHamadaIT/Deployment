# models.py
import os
import tensorflow as tf
import numpy as np

def predict(image_path):
    pass

# Define necessary functions for Model 1
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path, target_size=(256, 256)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['no', 'yes']
    pred_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return pred_class, confidence

# Load Model 1
model1 = tf.keras.models.load_model('E:/GP TEAM/gg/model/model2.h5')

# Load Model 2
model2 = tf.keras.models.load_model('E:/GP TEAM/gg/model/model2.h5')

# Function to check if the input image is likely a brain image
def is_brain_image(img_path):
    # Check if the file name contains keywords indicating it's a brain image
    brain_keywords = ['brain', 'tumor']
    if any(keyword in img_path.lower() for keyword in brain_keywords):
        return True
    
    # Check if the image dimensions are within a reasonable range for brain images
    img = tf.keras.preprocessing.image.load_img(img_path)
    img_size = img.size
    min_size = 100  # Adjust this threshold based on your dataset
    max_size = 500  # Adjust this threshold based on your dataset
    if min_size <= min(img_size) <= max_size:
        return True
    
    return False

# Function to integrate both models
def integrate_models(image_path):
    if not is_brain_image(image_path):
        return "Error: Input image does not seem to be a brain image. Please try again with a valid brain image."
    
    # Predict using Model 1
    img_array = preprocess_image(image_path)
    pred_class, confidence = predict_image(model1, img_array)
    
    if pred_class == 'yes':
        # If Model 1 predicts 'yes', pass the image to Model 2
        new_img_array = preprocess_image(image_path, target_size=(256, 256))
        predictions = model2.predict(new_img_array)
        class_labels = ['glioma_tumor', 'meningioma_tumor', 'you have tumor but we can not classify it ', 'pituitary_tumor']
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class = class_labels[predicted_class_index[0]]
        return "Predicted class from Model 2: {}".format(predicted_class)
    else:
        # If Model 1 predicts 'no', print 'no' along with confidence
        return "Predicted class from Model 1: {}\nConfidence (%): {}".format(pred_class, confidence)

# Call the integration function
#return integrate_models(image_path)

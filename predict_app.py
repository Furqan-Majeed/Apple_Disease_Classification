import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
# Make sure this matches the file name of your BEST training run
model_path = 'apple_disease_model_v1.keras' 

# Replace this with the actual path to an image you want to test
image_path_to_test = '/Users/furqanmajeed/Documents/apple_classification/AppleLeafDisease/All/Healthy/0cfa2ddf-56d9-4f62-89a4-f74e220f1859___RS_HL 6196.JPG'

# --- 2. LOAD MODEL ---
if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found. Did you rename it?")
    exit()

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)
class_names = ['Altenaria_Leaf_Spot', 'Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'Healthy', 'Powdery_Mildew']

# --- 3. PREDICT ---
print(f"Analyzing image: {image_path_to_test}")

try:
    img = tf.keras.utils.load_img(image_path_to_test, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(score)

    print(f"\n✅ RESULT: {predicted_class}")
    print(f"📊 CONFIDENCE: {confidence:.2f}%")
    
    # Show the image
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.1f}%)")
    plt.axis("off")
    plt.show()

except Exception as e:
    print(f"\n❌ ERROR: Could not read image. {e}")
    print("Check that your 'image_path_to_test' is correct.")
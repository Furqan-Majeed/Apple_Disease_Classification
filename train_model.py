import os
import logging
import warnings
import numpy as np # Needed for matrix
import matplotlib.pyplot as plt
import seaborn as sns # For nice matrix plotting
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. SUPPRESS WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- 2. IMPORTS ---
import tensorflow as tf
from tensorflow.keras import models, layers

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Detected: {len(gpus)}")

# --- 3. CONFIGURATION ---
BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 10

# UPDATE THIS PATH
dataset_path = '/Users/furqanmajeed/Documents/apple_classification/AppleLeafDisease' 

# --- 4. LOAD DATA (TRAINING) ---
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_path, 'Train'),
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# --- LOAD DATA (VALIDATION - FOR TRAINING) ---
print("Loading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_path, 'Val'),
    shuffle=True, # Shuffle is GOOD for training
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Save class names for later
class_names = train_ds.class_names
print(f"Classes: {class_names}")

# Optimize performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 5. BUILD MODEL ---
# --- 5. BUILD MODEL (The Correct Updated Version) ---
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 6

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=input_shape),
    
    # --- 1. AUGMENTATION (Crucial for your small dataset) ---
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1), 
    
    # --- CONV BLOCKS ---
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64,  kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64,  kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # --- 2. DROPOUT (Prevents the "Crash") ---
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.summary()

# --- 6. COMPILE & TRAIN ---
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS
)

# --- 7. EVALUATE ON VALIDATION SET ---
print("\n--- Evaluating Final Model ---")
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

# --- 8. PLOT ACCURACY & LOSS ---
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training vs Validation Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_results(history)

# --- 9. CONFUSION MATRIX (THE "CORRELATION" MATRIX) ---
print("\n--- Generating Confusion Matrix ---")

# We need a validation set WITHOUT shuffling to match predictions to labels correctly
val_ds_no_shuffle = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_path, 'Val'),
    shuffle=False, # CRITICAL: Must be false for confusion matrix
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Get Predictions
y_pred_probs = model.predict(val_ds_no_shuffle)
y_pred = np.argmax(y_pred_probs, axis=1)

# Get True Labels
y_true = np.concatenate([y for x, y in val_ds_no_shuffle], axis=0)

# Create Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print Classification Report (Precision/Recall)
print(classification_report(y_true, y_pred, target_names=class_names))

# --- 10. SAVE THE MODEL ---
model_version = 1
model.save(f"apple_disease_model_v{model_version}.keras")
print(f"Model saved successfully as apple_disease_model_v{model_version}.keras")
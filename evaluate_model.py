import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
val_dir = "processed_data/val"  # path to your validation folder
model_path = "deepfake_detection_model.h5"

# Image properties
img_size = (96, 96)
batch_size = 32

# Load the saved model
model = load_model(model_path)
print("✅ Model loaded successfully.")

# Set up validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Important for accurate evaluation
)

# Evaluate the model
loss, acc = model.evaluate(val_generator)
print(f"\n🔍 Validation Accuracy: {acc:.2%}")
print(f"🔍 Validation Loss: {loss:.4f}")

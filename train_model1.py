import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shutil

# Paths
csv_path = "train.csv"
base_dir = r"C:\Users\GADADASU GANESH\Downloads\real_vs_fake\real-vs-fake"
working_dir = "processed_data"
train_dir = os.path.join(working_dir, "train")
val_dir = os.path.join(working_dir, "val")

# Create directories
for split_dir in [train_dir, val_dir]:
    for class_dir in ["REAL", "FAKE"]:
        os.makedirs(os.path.join(split_dir, class_dir), exist_ok=True)

# Load and split CSV
df = pd.read_csv(csv_path)
df['label_str'] = df['label_str'].str.upper()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Copy images into structured folders
def copy_images(subset_df, subset_dir):
    for _, row in subset_df.iterrows():
        label = row['label_str']
        src = os.path.join(base_dir, row['path'])
        dst = os.path.join(subset_dir, label, os.path.basename(row['path']))
        if os.path.exists(src):
            shutil.copy(src, dst)

print("Copying training images...")
copy_images(train_df, train_dir)
print("Copying validation images...")
copy_images(val_df, val_dir)

# Set up ImageDataGenerator
img_size = (96, 96)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("deepfake_detection_model.h5")

# Evaluate it
loss, acc = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {acc:.2%}, Loss: {loss:.4f}")


# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the trained model
model.save("deepfake_detection_model.h5")

print("✅ Model training complete and saved.")

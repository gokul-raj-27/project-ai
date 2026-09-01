import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# SETTINGS
# ==========================================
DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ==========================================
# CHECK DATASET
# ==========================================
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

# ==========================================
# DATA AUGMENTATION
# ==========================================
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    shear_range=0.10
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================================
# CLASS MAPPING
# ==========================================
class_names = [
    name
    for name, index in sorted(
        train_data.class_indices.items(),
        key=lambda x: x[1]
    )
]

print("\n===================================")
print("CLASS MAPPING")
print("===================================")

for index, name in enumerate(class_names):
    print(f"{index} -> {name}")

with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("\nNumber of classes:", len(class_names))
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)

# ==========================================
# MOBILE NET V2
# ==========================================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# ==========================================
# NEW CLASSIFICATION HEAD
# ==========================================
model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# ==========================================
# COMPILE
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# CALLBACKS
# ==========================================
callbacks = [

    EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "best_isl_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ==========================================
# MODEL SUMMARY
# ==========================================
model.summary()

# ==========================================
# TRAIN
# ==========================================
print("\n===================================")
print("STARTING TRAINING")
print("===================================\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# SAVE FINAL MODEL
# ==========================================
model.save("final_isl_model.keras")

print("\n===================================")
print("TRAINING COMPLETE")
print("===================================")
print("Best model  : best_isl_model.keras")
print("Final model : final_isl_model.keras")
print("Classes     : class_names.json")
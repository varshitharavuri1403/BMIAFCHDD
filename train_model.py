import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "Coronary Artery Disease",
    "Cardiomyopathy",
    "Heart Failure",
    "Myocardial Infarction",
    "Arrhythmia",
    "Valvular Heart Disease",
    "Congenital Heart Disease",
    "Normal / No Abnormality",
]

IMAGE_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = os.path.join("models", "heart_cnn.keras")


def create_synthetic_dataset(num_samples_per_class=96, seed=42):
    rng = np.random.default_rng(seed)
    x = []
    y = []

    for class_index in range(NUM_CLASSES):
        base_color = np.array([
            0.1 + class_index * 0.08,
            0.2 + (class_index % 3) * 0.11,
            0.3 + (class_index % 4) * 0.09,
        ], dtype=np.float32)

        for _ in range(num_samples_per_class):
            image = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)
            # base gradient background
            for channel in range(3):
                image[:, :, channel] = base_color[channel]

            # add random tissue-like patterns
            noise = rng.normal(0.0, 0.12, size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

            # create a synthetic circular heart-like region
            yy, xx = np.ogrid[:IMAGE_SIZE[0], :IMAGE_SIZE[1]]
            center_y = IMAGE_SIZE[0] / 2 + rng.normal(0, 8)
            center_x = IMAGE_SIZE[1] / 2 + rng.normal(0, 8)
            radius = 70 + rng.normal(0, 8)
            mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius ** 2
            image[mask] = np.clip(image[mask] + np.array([0.15, 0.08, 0.12], dtype=np.float32), 0, 1)

            x.append(image)
            y.append(class_index)

    x = np.stack(x, axis=0)
    y = np.asarray(y, dtype=np.int32)
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    return x, y


def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    x_train, y_train = create_synthetic_dataset(num_samples_per_class=80)
    x_val, y_val = create_synthetic_dataset(num_samples_per_class=20, seed=123)

    model = build_model()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=16,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

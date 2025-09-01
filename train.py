import os
import json
import argparse
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
import pandas as pd

def build_dataset(train_dir, val_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_ds.class_names
    # Cache + prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model(num_classes, img_size=224, backbone="efficientnet"):
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augment")

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    if backbone.lower() == "mobilenet":
        base = MobileNetV2(include_top=False, input_tensor=x, weights="imagenet", pooling="avg")
    else:
        base = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet", pooling="avg")

    base.trainable = False  # freeze backbone initially

    x = layers.Dropout(0.25)(base.output)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="alz_classifier")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path containing train/ and val/ folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="efficientnet", choices=["efficientnet", "mobilenet"])
    parser.add_argument("--out_dir", type=str, default="model")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Loading data from {train_dir} and {val_dir}")
    train_ds, val_ds, class_names = build_dataset(train_dir, val_dir, args.img_size, args.batch_size)
    num_classes = len(class_names)
    print("[INFO] Classes:", class_names)

    print("[INFO] Building model...")
    model = build_model(num_classes, img_size=args.img_size, backbone=args.backbone)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, "alz_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    ]

    print("[INFO] Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Unfreeze for fine-tuning (optional small step)
    model.layers[2].trainable = True  # base model inside our stack
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("[INFO] Fine-tuning...")
    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=2, callbacks=callbacks)

    # Save final model
    model.save(os.path.join(args.out_dir, "alz_model.keras"))
    print("[INFO] Saved model to", os.path.join(args.out_dir, "alz_model.keras"))

    # Save labels
    labels_path = os.path.join(args.out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print("[INFO] Saved labels to", labels_path)

    # Save training history
    hist = history.history
    for k, v in history_ft.history.items():
        hist[f"ft_{k}"] = v
    df = pd.DataFrame(hist)
    df["timestamp"] = datetime.now().isoformat()
    hist_path = os.path.join(args.out_dir, "training_history.csv")
    df.to_csv(hist_path, index=False)
    print("[INFO] Saved training history to", hist_path)

if __name__ == "__main__":
    main()

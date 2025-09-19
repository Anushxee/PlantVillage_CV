import tensorflow as tf
import matplotlib.pyplot as plt

img_size = (224,224)
batch_size = 32
data_dir = "dataset"

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Model: MobileNetV2
base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                         include_top=False,
                                         weights="imagenet",
                                         pooling="avg")
base.trainable = False  # freeze base

x = tf.keras.layers.Dropout(0.3)(base.output)
out = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs=base.input, outputs=out)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save model
model.save("model/plant_disease.h5")

# Save class names
with open("model/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

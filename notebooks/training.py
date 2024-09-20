import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Define parameters
img_size = 224
channels = 3
batch_size = 32
epochs = 50
weight_decay = 1e-4
class_names = ['Early_blight', 'Healthy', 'Late_blight']

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Resize and rescale
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(img_size, img_size),
    layers.Rescaling(1.0 / 255)
])

# Define the model using Input layer for input shape
model = models.Sequential([
    tf.keras.Input(shape=(img_size, img_size, channels)),
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPool2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPool2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPool2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPool2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Dense(len(class_names), activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Load the dataset from the given directory
dataset_path = '/home/rahul/Documents/Potato-Disease-Classification/notebooks/dataset'

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='int'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode='int'
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {test_acc}")
print(f"Validation Loss: {test_loss}")

# Save the model in the 'models' folder (one step back)
model.save('/home/rahul/Documents/Potato-Disease-Classification/models/classifier_model.keras')

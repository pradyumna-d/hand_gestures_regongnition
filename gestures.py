import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import os

# Step 1: Define the dataset and data augmentation parameters
dataset_dir = os.path.expanduser('~/Desktop/ssd model')
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')
image_width, image_height = 224, 224
batch_size = 32

# Step 2: Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)
val_dataset = val_datagen.flow_from_directory(
    val_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)
test_dataset = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 3: Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 4: Define the loss function and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Step 5: Compile the model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Step 6: Train the model
model.fit(train_dataset, epochs=100, validation_data=val_dataset)

# Step 7: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')

# Step 8: Save the trained model
model.save(os.path.join(dataset_dir, 'model'))

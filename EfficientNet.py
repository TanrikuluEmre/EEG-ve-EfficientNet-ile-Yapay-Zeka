# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# # Define the path to the directory containing the spectrogram images
# path = "spectrograms\\"

# # Load the EfficientNet model
# model = tf.keras.applications.EfficientNetB0(weights='imagenet')

# # Loop over all .png files in the directory
# for filename in os.listdir(path):
#     if filename.endswith(".png"):
#         # Load the image and preprocess it
#         img = load_img(os.path.join(path, filename), target_size=(224, 224))
#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         # Use the model to predict the class probabilities
#         preds = model.predict(x)
#         # Decode the predictions into human-readable labels
#         labels = decode_predictions(preds, top=3)[0]
#         # Print the predicted labels
#         print("Predictions for", filename, ":")
#         for label in labels:
#             print(label[1], "-", label[2])

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0

# Define the path to the directory containing the spectrogram images
path = "spectrograms/"

# Define the image size and batch size
img_size = (224, 224)
batch_size = 32

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')
val_generator = train_datagen.flow_from_directory(
    path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Load the EfficientNet model
model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in model.layers:
    layer.trainable = False

# Add new classification layers
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

# Create the new model
model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)




import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from efficientnet.tfkeras import EfficientNetB0


input_shape = (640, 480, 3)
num_classes = 2
batch_size = 32
epochs = 10
learning_rate = 0.001

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='train\\',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    directory='test\\',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Modeli tanımlayın
model = Sequential()

# EfficientNetB0 modelini ekleyin
model.add(EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape))

# Flatten katmanını ekleyin
model.add(Flatten())

# Yoğun katmanı ekleyin
model.add(Dense(256, activation='relu'))

# Dropout katmanını ekleyin
model.add(Dropout(0.5))

# Çıkış katmanını ekleyin
model.add(Dense(num_classes, activation='softmax'))

# Modeli derleyin
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[EarlyStopping(patience=3)])

# Modeli değerlendirin
score = model.evaluate(test_generator)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

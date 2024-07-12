import os
import cv2  # Computer Vision library, used to download images
import numpy as np  # Working with numpy arrays
import matplotlib.pyplot as plt  # Used for visualization of digits
import tensorflow as tf  # Framework used for machine learning part, for training

mnist = tf.keras.datasets.mnist  # Loading the dataset directly from TensorFlow

# 1. Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess the data (normalize)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# 3. Build the model (using CNNs and max pooling)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Train the model
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=10)

# 6. Save the trained model
model.save('handwritten_cnn.h5')

# 7. Load the trained model
model = tf.keras.models.load_model('handwritten_cnn.h5')

# 8. Evaluate the model
loss, accuracy = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# 9. Make predictions on custom images
image_number = 0
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28)) 
        img = np.invert(img) 
        img = img.reshape(-1, 28, 28, 1)  
        prediction = model.predict(img)
        print(f"Prediction for digit {image_number}: {np.argmax(prediction)}")
        plt.imshow(img.reshape(28, 28), cmap='binary')
        plt.show()
    except Exception as e:
        print("Error:", e)
    finally:
        image_number += 1

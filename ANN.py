import os
import pandas as pd
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import struct
import cv2
os.environ["PYTHONIOENCODING"] = "utf-8"

def load_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
x_train=load_idx('C:/Users/Mitesh/Desktop/jupyter project/Image processing/Perceptron/archive (3)/train-images.idx3-ubyte')
y_train=load_idx('C:/Users/Mitesh/Desktop/jupyter project/Image processing/Perceptron/archive (3)/train-labels.idx1-ubyte')

x_test=load_idx('C:/Users/Mitesh/Desktop/jupyter project/Image processing/Perceptron/archive (3)/t10k-images.idx3-ubyte')
y_test=load_idx('C:/Users/Mitesh/Desktop/jupyter project/Image processing/Perceptron/archive (3)/t10k-labels.idx1-ubyte')

def plot_images(images, labels, num_images=5):
    # Create a figure for displaying images
    plt.figure(figsize=(10, 2))
    
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)  # Create a subplot for each image
        plt.imshow(images[i], cmap='gray')  # Display the image in grayscale
        plt.title(f'Label: {labels[i]}')  # Set the title as the corresponding label
        plt.axis('off')  # Turn off the axis for a cleaner visualization
    plt.show()  # Show the plot with the selected images and labels
# Example: Plot 5 images and their corresponding labels    
plot_images(x_train, y_train, num_images=6)

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(512, activation ='relu'))
model.add(tf.keras.layers.Dense(256, activation ='relu'))
model.add(tf.keras.layers.Dense(128, activation ='relu'))
model.add(tf.keras.layers.Dense(10, activation ='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10,verbose=2)
model.save('my_model.keras')

model1=tf.keras.models.load_model('my_model.keras')

loss,accuracy = model1.evaluate(x_test,y_test,verbose=2)

image_number=1
while os.path.isfile(f"my numbers/digit {image_number}.png"):
    img=cv2.imread(f"my numbers/digit {image_number}.png")[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img,verbose=0)
    print(f"this image is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    image_number += 1

    
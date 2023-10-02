import os
import tensorflow as tf  # machine learning
import numpy as np  # numpy arrays
import matplotlib.pyplot as plt  # for visualization
import cv2  # computer vision (load images, process images)

# load dataset
mnist = tf.keras.datasets.mnist

# labelled data (data that we already have classified) is split into training and testing
# x is image, y is label
# load data from mnist dataset (two tuples),
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# want 80% of data to be training data, 20% to be testing data (.load_data() splits it for us)

# preprocessing
# normalize data (scale data to be between 0 and 1 from 0 to 255)
x_train = tf.keras.utils.normalize(x_train, axis=1)  # normalize training data
x_test = tf.keras.utils.normalize(x_test, axis=1)  # normalize testing data

"""
# #build neural network model
# model = tf.keras.models.Sequential() #Sequential() is the simplest model, a feed-forward model

# #input layer (turns 28x28 images into one big line of pixels basically)
# model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #dense layer is a fully connected layer, 128 neurons, activation function is relu
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #another dense layer
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #10 neurons, 10 classes (0-9), activation function is softmax
# #softmax is a probability distribution function, gives probability of each class

# #compile model, optimizer is how model is updated based on data it sees and its loss function, loss function is how well model did on training, metrics is what we want to track
# model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3) #train model, epochs is how many times model sees same information

# model.save('digits.model')
"""

# load model
model = tf.keras.models.load_model('digits.model')

# test model
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)  # loss should be low, accuracy should be high
print(accuracy)

# make predictions

for files in os.listdir("testing data"):
    if files.endswith('.jpeg'):
        try:
            img = cv2.imread(os.path.join("testing data", files))[
                :, :, 0]  # read image, convert to grayscale
            # invert image, convert to numpy array
            img = np.invert(np.array([img]))
            prediction = model.predict(img)  # make prediction
            print("The number is probably", np.argmax(prediction), " ",
                  "the number is actually", files)  # print prediction
            plt.imshow(img[0], cmap=plt.cm.binary)  # show image
            plt.show()
        except:
            print("Error reading image")

'''Import TensorFlow into the program'''

import tensorflow as tf
#print("TensorFlow version:", tf.__version__)

'''Load and prepare the MNIST dataset. The pixel values of the images range from 0 through 255.
Scale these values to a range of 0 to 1 by dividing the values by 255.0.
This also converts the sample data from integers to floating-point numbers:'''

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


'''BUILD A MACHINE LEARNING MODEL'''

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

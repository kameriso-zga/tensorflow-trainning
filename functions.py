import tensorflow as tf
import matplotlib.pyplot as plt

#Loads mnist data set
def load_data():
    return tf.keras.datasets.mnist.load_data()

def print_images(input_images, output_images, num_of_images):
    #Resizes plot figure
    plt.figure(figsize=(16,10))

    #Prints number of images passed using imshow and show functions
    for index in range(1, num_of_images + 1):
        plt.subplot(5, 10, index).set_title(f'{output_images[index]}')
        plt.axis('off')
        plt.imshow(input_images[index], cmap='gray')

    plt.show()

#Function that processes data to prepare it for Keras
def data_preprocessor(training_input, test_input):
    #Number of rows and columns
    image_rows = training_input[0].shape[0]
    image_cols = test_input[0].shape[1]

    input_shape = (image_rows, image_cols, 1)

    #Putting data in the right dimensions for Keras
    #Add 4th dimension to data
    training_input = training_input.reshape(training_input.shape[0], image_rows, image_cols, 1)
    test_input = test_input.reshape(test_input.shape[0], image_rows, image_cols, 1)

    #Change image to float32
    training_input = training_input.astype('float32')
    test_input = test_input.astype('float32')

    #Change data range from 0-255 to 0-1
    training_input /= 255.0
    test_input /= 255

    return training_input, test_input, input_shape

#One hot encodes labels using "to_catagorical" from "tensorflow.keras.utils"
def one_hot_encoding_labels(training_output, test_output):
    training_output = tf.keras.utils.to_categorical(training_output)
    test_output = tf.keras.utils.to_categorical(test_output)

    return training_output, test_output




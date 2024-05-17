import functions as func
from tensorflow import keras

(preprocessed_x_train, prehotencoded_y_trained), (preprocessed_x_test, prehotencoded_y_test) = func.load_data()

x_train, x_test, input_shape = func.data_preprocessor(preprocessed_x_train, preprocessed_x_test)
y_train, y_test = func.one_hot_encoding_labels(prehotencoded_y_trained, prehotencoded_y_test)

#Defining number of classes for our one hot encoded matrix and number of pixels in each image
num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

#Creates model
model = keras.Sequential()

#Adds first convolutional layer with a filter size of 32
#Reduces layer size to 26 x 26 x 32
model.add(keras.layers.Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))

#Second convolutional layer with filter size of 64
#Reduces our layer size to 24 x 24 x 64
model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))

#MaxPooling with a kernel size of 2 x 2
#Reduces size tto 12 x 12 x 64
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

#Flatten before dense layer to reshape the tensor to have the shape that is equal to number of elements
#contained in the tensor as in CNN
model.add(keras.layers.Flatten())

#Fully connected dense layer of size 1 x 128
model.add(keras.layers.Dense(128, activation = 'relu'))

#Create dense layer with number of classes
model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

#Compile model, this creates an object that stores the model

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.SGD(0.01),
              metrics = ['accuracy'])

#Training the model
batch_size = 128
epochs = 25

#Store results for ploting
# history = model.fit(x_train,
#                     y_train,
#                     batch_size = batch_size,
#                     epochs = epochs,
#                     verbose = 1,
#                     validation_data = (x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose = 0)

#Save model
model.save("mnist_simple_cnn_25_epochs.keras")
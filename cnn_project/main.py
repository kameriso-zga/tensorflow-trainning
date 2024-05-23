import model
from tensorflow import keras
import numpy as np

classifier = keras.models.load_model("mnist_simple_cnn_25_epochs.keras")
pred = np.argmax(classifier.predict(model.x_test), axis=-1)

print(pred)
print(pred.shape)
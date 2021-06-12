from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

FILTERS = 32        # number of filters the convolutional layer will learn
POOL_SIZE = (2, 2)  # 2 X 2 pixels for size of pooling operation
K_SIZE1 = (5, 5)    # kernel size for the first Conv2D layer
K_SIZE2 = (3, 3)    # kernel size for the second Conv2D layer
DENSE_LAYER_NODES = 64
DROPOUT_RATE = 0.5


class SeqNet:
    @staticmethod
    def make(width, height, depth, classes):
        model = Sequential()

        # Down sampling: Conv2D --> Nonlinearity --> Pooling Layer
        model.add(Conv2D(FILTERS, K_SIZE1, padding="same", input_shape=(height, width, depth)))
        model.add(Activation("relu"))   # add rectified linear unit activation function
        model.add(MaxPooling2D(pool_size=POOL_SIZE))    # use max pooling to reduce spatial dimensions of output volume

        # Second Conv2D --> Nonlinearity --> Pooling Layer set
        model.add(Conv2D(FILTERS, K_SIZE2, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=POOL_SIZE))

        # Dropout NN Layer 1 (regularization method; randomly drops nodes)
        model.add(Flatten())
        model.add(Dense(DENSE_LAYER_NODES))
        model.add(Activation("relu"))
        model.add(Dropout(DROPOUT_RATE))

        # Dropout NN Layer 2, without flattening
        model.add(Dense(DENSE_LAYER_NODES))
        model.add(Activation("relu"))
        model.add(Dropout(DROPOUT_RATE))

        # softmax classifier --> yields actual probability scores for each class label
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model



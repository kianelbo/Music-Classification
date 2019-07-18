from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
import pickle


class SpectrogramBasedModel:
    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = load_model(model_path)
        else:
            self.model = None

    def train(self, X_pickle, y_pickle):
        X = pickle.load(open(X_pickle, 'rb'))
        y = pickle.load(open(y_pickle, 'rb'))
        y = to_categorical(y)
        print('read pickles')

        self.model = Sequential()
        self.model.add(Conv2D(64, (2, 2), input_shape=X.shape[1:]))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (2, 2)))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (2, 2)))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (2, 2)))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('elu'))
        self.model.add(Dropout(.5))

        self.model.add(Dense(5))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(X, y, batch_size=32, epochs=3, validation_split=0.2)
        print('training done')

        self.model.save('save/sb.model')

    def predict(self):
        pass

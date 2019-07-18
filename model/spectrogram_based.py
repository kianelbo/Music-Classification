import pickle
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D


class SpectrogramBasedModel:
    def __init__(self, model_path=None):
        self.encoder = pickle.load(open('save/encoder.pickle', 'rb'))
        if model_path is not None:
            self.model = load_model(model_path)
        else:
            self.model = None

    def train(self, X_pickle, y_pickle):
        X = pickle.load(open(X_pickle, 'rb'))
        y = pickle.load(open(y_pickle, 'rb'))

        n_classes = len(self.encoder.classes_)

        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), input_shape=X.shape[1:]))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (2, 2)))
        self.model.add(Activation('elu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(.2))
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('elu'))

        self.model.add(Dense(n_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(X, y, batch_size=32, epochs=5, validation_split=0.2)
        print('training done')

        self.model.save('save/sb.model')

    def predict(self, spectrogram_slices):
        slices = np.array(spectrogram_slices).reshape(-1, 128, 128, 1)
        prediction = self.model.predict(slices)
        prediction = np.argmax(prediction, axis=1)
        print(self.encoder.inverse_transform(prediction))

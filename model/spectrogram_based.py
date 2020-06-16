import pickle
import numpy as np
from tensorflow.keras import models, Sequential, callbacks
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


class SpectrogramBasedModel:
    def __init__(self, model_path=None):
        self.name = 'spectrogram_based_model'
        self.log_dir = "logs/fit/" + self.name
        self.encoder = pickle.load(open('save/encoder.pickle', 'rb'))
        if model_path is not None:
            self.model = models.load_model(model_path)
        else:
            self.model = None

    def train(self, X_pickle, y_pickle):
        X = pickle.load(open(X_pickle, 'rb'))
        y = pickle.load(open(y_pickle, 'rb'))
        y = np.asarray(y)

        n_classes = len(self.encoder.classes_)

        self.model = Sequential()
        self.model.add(Conv2D(32, (2, 2), input_shape=X.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(.2))
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        self.model.add(Dense(n_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        tb_callback = callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        self.model.fit(X, y, batch_size=32, epochs=8, validation_split=0.2, callbacks=[tb_callback])
        print('training done\n')
        print(self.model.summary())

        self.model.save('./save/sb_model')

    def predict(self, spectrogram_slices):
        return self.model.predict(spectrogram_slices)

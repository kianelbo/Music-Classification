import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, Sequential, callbacks


class FeaturesBasedModel:
    def __init__(self, model_path=None):
        self.name = 'features_based_model'
        self.log_dir = "logs/fit/" + self.name
        self.encoder = pickle.load(open('save/encoder.pickle', 'rb'))
        if model_path is not None:
            self.model = models.load_model(model_path)
            self.scaler = pickle.load(open('save/f_scaler.pickle', 'rb'))
        else:
            self.model = None
            self.scaler = StandardScaler()

    def train(self, csv_file):
        data = pd.read_csv(csv_file, encoding='ISO-8859-1')

        data.drop(['title'], axis=1, inplace=True)

        # obtaining target column
        genre_column = data['genre']
        y = self.encoder.transform(genre_column)
        n_classes = len(self.encoder.classes_)

        # obtaining feature columns and scaler
        X = np.array(data.drop(['genre'], axis=1), dtype=float)
        X = self.scaler.fit_transform(X)
        with open('save/f_scaler.pickle', 'wb') as pickle_out:
            pickle.dump(self.scaler, pickle_out)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model = Sequential()
        self.model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(n_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        tb_callback = callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        self.model.fit(X_train, y_train, epochs=20, callbacks=[tb_callback])
        print('training done\n')

        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print('test_acc: ', test_acc)
        print(self.model.summary())

        self.model.save('./save/fb_model')

    def predict(self, features_set):
        scaled_features = self.scaler.transform(np.array(features_set, dtype=float))
        return self.model.predict(scaled_features)

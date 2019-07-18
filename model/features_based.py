import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow.python.keras as keras


class FeaturesBasedModel:
    def __init__(self, model_path=None):
        if model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            self.model = None

    def train(self, csv_file):
        data = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print(data.head())
        print(data.shape)

        data.drop(['title'], axis=1, inplace=True)
        genre_list = data.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(genre_list)

        X = StandardScaler().fit_transform(np.array(data.iloc[:, :-1], dtype=float))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit(X_train, y_train, epochs=30)
        print('training done')

        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print('test_acc: ', test_acc)

        self.model.save('save/fb.model')

    def predict(self):
        pass

# sample = [create_features('../../x.wav').split(' ')]
# scaled_sample = scaler.fit_transform(np.array(sample, dtype=float))
# prediction = np.argmax(model.predict(scaled_sample))
# print(prediction)
#
# print(encoder.inverse_transform([prediction]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow.python.keras as keras

from preprocessing.extract_features import create_features

data = pd.read_csv('../data.csv')
print(data.head())
print(data.shape)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()

y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = keras.models.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=30)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('test_acc: ', test_acc)

model.save('first.model')

# sample = [create_features('../../x.wav').split(' ')]
# scaled_sample = scaler.fit_transform(np.array(sample, dtype=float))
# prediction = np.argmax(model.predict(scaled_sample))
# print(prediction)
#
# print(encoder.inverse_transform([prediction]))

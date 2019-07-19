import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from features.extract_features import create_features, build_features_csv
from features.extract_spectrogram import generate_spectrogram_slices, build_spectrogram_dataset


def extract(filename):
    features_set = [fs.split(' ') for fs in create_features(filename)]
    spectrogram_slices = np.array(generate_spectrogram_slices(filename)).reshape(-1, 128, 128, 1)
    return features_set, spectrogram_slices


def make_training_data(dataset_directory):
    print('building the encoder')
    encoder = LabelEncoder()
    encoder.fit(os.listdir(dataset_directory))
    pickle_out = open('save/encoder.pickle', 'wb')
    pickle.dump(encoder, pickle_out)
    pickle_out.close()

    # print('gathering features...')
    # build_features_csv(dataset_directory)
    # print('done.')

    # print('gathering spectrogram slices...')
    # build_spectrogram_dataset(dataset_directory)
    # print('done.')

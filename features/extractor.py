from features.extract_features import create_features, build_features_csv
from features.extract_spectrogram import generate_spectrogram_slices, build_spectrogram_dataset


def extract(filename):
    features_set = create_features(filename)
    spectrogram_slices = generate_spectrogram_slices(filename)
    return features_set, spectrogram_slices


def make_training_data(dataset_directory):
    # print('gathering features...')
    # build_features_csv(dataset_directory)
    # print('done.')

    print('gathering spectrogram slices...')
    build_spectrogram_dataset(dataset_directory)
    print('done.')

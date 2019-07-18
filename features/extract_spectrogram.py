import os
import pickle
import subprocess
import numpy as np
from random import shuffle
from PIL import Image

save_dir = 'save/'


def generate_spectrogram_slices(song):
    subprocess.run(
        ['sox', song, '-n', 'remix', '1,2', 'trim', '60', '60', 'spectrogram', '-Y', '200', '-X', '50',
         '-m', '-r', '-o', save_dir + 'spec.png'])

    img = Image.open(save_dir + 'spec.png')
    n_samples = int(img.size[0] / 128)
    slices = []
    for i in range(n_samples):
        start_pixel = i * 128
        cropped_slice = img.crop((start_pixel, 1, start_pixel + 128, 128 + 1))
        slice_data = np.asarray(cropped_slice, dtype=np.uint8).reshape(128, 128, 1)
        slice_data = slice_data / 255.
        slices.append(slice_data)

    os.remove(save_dir + 'spec.png')

    return slices


def build_spectrogram_dataset(dataset_path):
    training_data = []
    print('gathering spectrogram from files...')
    for genre_num, genre in enumerate(os.listdir(dataset_path)):
        print(genre)
        genre_folder = f'{dataset_path}/{genre}/'
        for filename in os.listdir(genre_folder):
            slices = generate_spectrogram_slices(f'{genre_folder}/{filename}')
            for s in slices:
                training_data.append([s, genre_num])

    shuffle(training_data)
    print('shuffle done')

    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 128, 128, 1)

    print('writing spectrogram data pickles...')
    pickle_out = open(save_dir + 'X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open(save_dir + 'y.pickle', 'wb')
    pickle.dump(y, pickle_out)
    pickle_out.close()
    print('done writing pickles.')

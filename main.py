import argparse
from features.extractor import make_training_data, extract
from model.ensemble import aggregate
from model.features_based import FeaturesBasedModel
from model.spectrogram_based import SpectrogramBasedModel

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", action='store_true', help="training models")
ap.add_argument("-i", "--input", required=True, help="input path")
args = vars(ap.parse_args())

if args["train"]:  # train phase
    # reading dataset directory
    make_training_data(args["input"])
    # training features-based network
    model = FeaturesBasedModel()
    model.train('save/data.csv')
    # training spectrogram-based network
    model = SpectrogramBasedModel()
    model.train('save/X.pickle', 'save/y.pickle')
else:  # predict phase
    # extracting features from input file
    features_set, spectrogram_slices = extract(args["input"])
    # features-based model prediction
    model = FeaturesBasedModel('./save/fb_model')
    features_based_prediction = model.predict(features_set)
    # spectrogram-based model prediction
    model = SpectrogramBasedModel('./save/sb_model')
    spectrogram_based_prediction = model.predict(spectrogram_slices)
    # aggregating and displaying results
    aggregate(features_based_prediction, spectrogram_based_prediction)

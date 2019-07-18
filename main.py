from features.extractor import make_training_data, extract
from model.features_based import FeaturesBasedModel
from model.spectrogram_based import SpectrogramBasedModel

# make_training_data('D:/dataset3/')

model = FeaturesBasedModel('save/fb.model')
# model.train('save/data.csv')
features_set, _ = extract('D:/Hells Bells.mp3')
model.predict(features_set)
# SpectrogramBasedModel().train('save/X.pickle', 'save/y.pickle')

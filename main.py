from features.extractor import make_training_data
from model.features_based import FeaturesBasedModel
from model.spectrogram_based import SpectrogramBasedModel

# make_training_data('D:/dataset3/')

# FeaturesBasedModel().train('save/data.csv')
SpectrogramBasedModel().train('save/X.pickle', 'save/y.pickle')

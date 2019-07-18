from features.extractor import make_training_data, extract
from model.features_based import FeaturesBasedModel
from model.spectrogram_based import SpectrogramBasedModel

# make_training_data('D:/dataset3/')

model = FeaturesBasedModel()
model.train('save/data.csv')
features_set, _ = extract('D:/Blues Leave Me Alone.mp3')
model.predict(features_set)
# SpectrogramBasedModel().train('save/X.pickle', 'save/y.pickle')

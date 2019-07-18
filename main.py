from features.extractor import make_training_data, extract
from model.features_based import FeaturesBasedModel
from model.spectrogram_based import SpectrogramBasedModel

make_training_data('D:/dataset2/')
features_set, spectrogram_slices = extract('D:/Orion.mp3')

model = FeaturesBasedModel('save/fb.model')
# model.train('save/data.csv')
model.predict(features_set)

model = SpectrogramBasedModel()
model.train('save/X.pickle', 'save/y.pickle')
model.predict(spectrogram_slices)

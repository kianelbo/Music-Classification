import pickle

encoder = pickle.load(open('save/encoder.pickle', 'rb'))


def aggregate(features_based_prediction, spectrogram_based_prediction):
    print('features-based neural network results:\n', features_based_prediction)
    feat_mean = features_based_prediction.mean(axis=0)
    feat_prediction = feat_mean.argmax()
    feat_confidence = feat_mean[feat_prediction]
    for i, c in enumerate(encoder.classes_):
        print(f"{'+' if i == feat_prediction else '-'} {c}: {feat_mean[i]}")

    print('spectrogram-based neural network results:\n', spectrogram_based_prediction)
    spec_mean = spectrogram_based_prediction.mean(axis=0)
    spec_prediction = spec_mean.argmax()
    spec_confidence = spec_mean[spec_prediction]
    for i, c in enumerate(encoder.classes_):
        print(f"{'+' if i == spec_prediction else '-'} {c}: {spec_mean[i]}")

    if feat_confidence > spec_confidence:
        print('final verdict:', encoder.classes_[feat_prediction])
    else:
        print('final verdict:', encoder.classes_[spec_prediction])

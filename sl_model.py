#What else should we take in as input
def save_model(model, path='model/', epoch_num=0):
    model.save_weights(path + 'epoch_%s.h5' % epoch_num)


def load_model(path):
    model.load_weights(path)
    print("Loaded")
    return model

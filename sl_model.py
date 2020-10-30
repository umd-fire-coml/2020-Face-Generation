#What else should we take in as input
def save_model(epoch_num):
    checkpoint_path = "model/"
    self.generator.save_weights('models/epoch_%s.h5' % (epoch))


def load_model(path):
    model.load_weights(checkpoint_path)
    #what info do we need?
    model.evaluate(test_images,  test_labels, verbose=2)
    return model

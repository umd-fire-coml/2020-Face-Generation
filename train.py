import matplotlib.pyplot as plt

from image_sequence_gen import ImageSequenceGenerator
from fake_attribute_sequence_generator import FakeAttributeSequenceGenerator
from sl_model import save_model

def make_trainable(model, val):
    model.trainable = val
    for lay in model.layers:
        lay.trainable = val    

# train the generator and discriminator
def train(genModel, discModel, ganModel, dataset, numEpochs=10, batchSize=32, numBatch=10):
    # manually enumerate epochs
    imageData = ImageSequenceGenerator(dataset, batchSize, genModel, imageSize=(64,64))
    attrData = FakeAttributeSequenceGenerator(batchSize)
    for i in range(numEpochs):
        # enumerate batches over the training set
        for j in range(numBatch):
            # create training set for the discriminator
            x, y = imageData[j]
            # update discriminator model weights
            make_trainable(discModel, True)
            d_loss, _ = discModel.train_on_batch(x, y * 0.9)
            make_trainable(discModel, False)
            
            x, y = attrData[j]
            # update the generator via the discriminator's error
            g_loss = ganModel.train_on_batch(x, y)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, numBatch, d_loss, g_loss))
        save_model(discModel, "model/discModel/", epoch_num=i)
        save_model(genModel, "model/genModel/", epoch_num=i)
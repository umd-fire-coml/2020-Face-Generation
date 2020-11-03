import matplotlib.pyplot as plt

from dataGens import ImageDataGen, FakeAttributeDataGen

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=4):
    # plot images
    for i in range(15):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0])
    # save plot to file
    filename = 'trainingData/savedImages/generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, genaModel, discModel, dataset, n_samples=100):
    imageData = ImageDataGen(dataset, genaModel, n_samples*2, imageSize=(64,64))
    _, acc_real = discModel.evaluate(imageData[0][0][:n_samples], imageData[0][1][:n_samples], verbose=0)
    _, acc_fake = discModel.evaluate(imageData[0][0][n_samples:], imageData[0][1][n_samples:], verbose=0)

    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(imageData[0][0][n_samples:], epoch)
    # save the generator model tile file
    filename = 'trainingData/models/generator_model_%03d.h5' % (epoch + 1)
    genaModel.save(filename)
    

def make_trainable(model, val):
    model.trainable = val
    for lay in model.layers:
        lay.trainable = val

# train the generator and discriminator
def train(genModel, discModel, ganModel, dataset, numEpochs=10, batchSize=32, numBatch=10):
    # manually enumerate epochs
    imageData = ImageDataGen(dataset, genModel, batchSize, imageSize=(64,64))
    attrData = FakeAttributeDataGen(batchSize)
    for i in range(numEpochs):
        # enumerate batches over the training set
        for j in range(numBatch):
            # create training set for the discriminator
            X, Y = imageData[j]
            # update discriminator model weights
            make_trainable(discModel, True)
            d_loss, _ = discModel.train_on_batch(X, Y * 0.9)
            make_trainable(discModel, False)
            
            x, y = attrData[j]
            # update the generator via the discriminator's error
            g_eval = ganModel.test_on_batch(x, y)
            g_loss = ganModel.train_on_batch(x, y, reset_metrics=True)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.10f, g=%.10f, g_eval=%.10f' % (i+1, j+1, numBatch, d_loss, g_loss, g_eval))
        # evaluate the model performance, sometimes
        if (i+1) % 1 == 0:
            summarize_performance(i, genModel, discModel, dataset)
            
            
            
            
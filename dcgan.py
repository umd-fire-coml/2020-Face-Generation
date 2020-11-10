from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np

from dataGens import ImageSequenceGenerator, FakeAttributeSequenceGenerator

class DCGAN():
    def __init__(self, dataset, path='', img_rows=64, img_cols=64, channels=3, latent_dim=41, gf=64, df=64):
        # Save parameters
        self.dataset = dataset
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.gf = gf
        self.df = df

        dOptimizer = Adam(0.0001, 0.5)
        gOptimizer = Adam(0.0002, 0.5)
        
        ### Generator Model ###
        # Build the generator
        self.generator = self.build_generator()

        ### Discriminator Model ###
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=dOptimizer,
            metrics=['accuracy'])
        
        ### Combined Model: GAN ###
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=gOptimizer)
        
        
        
    def getBase(self):
        power = 0
        startR = self.img_rows
        startC = self.img_cols
        
        while (startR > 10 and startC > 10 and
               startR % 2 == 0 and startC % 2 == 0):
            power += 1
            startR /= 2
            startC /= 2
        return (int(startR), int(startC)), power

    def build_generator(self):
        start, power = self.getBase()
        
        def g_block(model, filters):
            # Convolution layers
            model.add(UpSampling2D())
            model.add(Conv2D(filters, kernel_size=5, padding="same"))
            
            # Batch normalization and Activation
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
        
        ### Build Model ###
        
        model = Sequential()
        
        # Input layers
        model.add(Dense(self.gf * (2 ** power) * start[0] * start[1], activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((start[0], start[1], self.gf * (2 ** power))))
        
        # Upscaling convolutionary layer blocks
        for f in range(power, 0, -1):
            g_block(model, self.gf * (2 ** f))

        # Format output
        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))

        # Display Model
        model.summary()

        # Join layers into a model
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    
    def build_discriminator(self):

        def d_block(model, filters, strides=2, bn=True, input_size=False):
            # Convolution layer with optional input shape
            if input_size:
                model.add(Conv2D(filters, kernel_size=4, strides=strides, input_shape=self.img_shape, padding="same"))
            else:
                model.add(Conv2D(filters, kernel_size=4, strides=strides, padding="same"))
            
            # Optional Batch normalization
#             if bn:
#                 model.add(BatchNormalization(momentum=0.8))
            
            # Activation and Dropout
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))

        ### Build Model ###
        
        model = Sequential()
        
        # Repetative inner layer blocks
        d_block(model, self.df * 1, bn=False, input_size=True) # Input layer
        d_block(model, self.df * 2)
        d_block(model, self.df * 4)
        d_block(model, self.df * 8, strides=1) # End
        
        # Format output
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        # Display Model
        model.summary()

        # Join layers into a model
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_num=100, batch_size=128, save_interval=5):

        # Load the datasets
        gDataGen = FakeAttributeSequenceGenerator(batch_size=batch_size, latent_dim=self.latent_dim)
        dDataGen = ImageSequenceGenerator(self.dataset, batch_size, self.generator, image_size=(self.img_rows,self.img_cols))

        for epoch in range(epochs):
            print ("----- Epoch %d -----\n" % (epoch))
            for batch in range(batch_num):
                ### Train Discriminator ###

                # Get Discriminator training data
                d_input, d_output = dDataGen[batch]

                # Train the Discriminator
                d_loss = self.discriminator.train_on_batch(d_input, d_output)

                ### Train Generator ###

                # Get Generator training data
                g_input, g_output = gDataGen[batch]

                # Train the Generator
                g_loss2 = self.combined.test_on_batch(g_input, g_output)
                g_loss = self.combined.train_on_batch(g_input, g_output)
                g_loss3 = self.combined.test_on_batch(g_input, g_output)

                # Print training results
                print ("Batch %d/%d - Discrim loss: %f, accuracy: %.2f%%\n             - Gener loss: %f, loss test: %.4f %.4f" %
                       (batch, batch_num, d_loss[0], 100*d_loss[1], g_loss, g_loss2, g_loss3))
            print("\n")
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                dDataGen.shuffle()
                self.save_imgs(epoch)
                self.save_model(self.combined, epoch_num=epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("sampleImgs/epoch_%d.png" % epoch)
        plt.close()
        
    def save_model(self, model, path='models/', epoch_num=0):
        model.save(path + 'epoch_%s.h5' % epoch_num)


    def load_model(self, model, path):
        model.load_weights(path)
        print("Loaded")
        
if __name__ == "__main__":
    ds = tfds.load('celeb_a', data_dir="./", download=False)
    dcgan = DCGAN(ds["train"], img_rows=64, img_cols=64, gf=32, df=32)
    dcgan.train(epochs=100, batch_num=100, batch_size=32, save_interval=1)
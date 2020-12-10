from tensorflow.keras.utils import Sequence
import numpy as np
import cv2

class ImageSequenceGenerator(Sequence):
    def __init__(self, dataset, size, model, image_size=(64,64)):
        self.data = dataset
        self.batch_size = int(size / 2) # Half will be real and half fake
        self.model = model # Saves generator model to load fake images
        self.image_size = image_size

    def __getitem__(self, idx):
        # Get and format real images
        images = self.data.skip(self.batch_size * idx).take(self.batch_size)
        real_image = [cv2.resize(np.array(i["image"]), dsize=self.image_size, interpolation=cv2.INTER_CUBIC) / 127.5 - 1 for
                      i in images]
        # Create fake images
        x_input = np.random.randint(2, size=(self.batch_size, 40))
        x_input = np.concatenate((x_input, np.random.uniform(size=(self.batch_size, 1))), axis=1)
        fake_image = np.reshape(self.model.predict(x_input),
                                (self.batch_size, self.image_size[1], self.image_size[0], 3))

        # Join image lists together
        batch_x = np.concatenate((real_image, fake_image))
        # Create truth values (0 means real and 1 means fake)
        batch_y = np.concatenate(
            (np.zeros((self.batch_size, 1), dtype=int), np.ones(((self.batch_size), 1), dtype=int)))

        return np.array(batch_x), np.array(batch_y)
    
    def shuffle(self):
        # Shuffles the dataset
        self.data.shuffle(10000)

class FakeAttributeSequenceGenerator(Sequence):
    def __init__(self, batch_size=64, latent_dim=41):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
    
    def __getitem__(self, idx):
        # Create fake attributes
        attrs = np.random.randint(2, size=(self.batch_size, self.latent_dim - 1))
        # Add a random value
        noise = np.random.uniform(size=(self.batch_size, 1))
        attrs = np.concatenate((attrs, noise), axis=1)
        
        # Fake truth values since the goal is to trick the discriminator
        truth = np.zeros((self.batch_size, 1), dtype = int) 
        
        return attrs, truth
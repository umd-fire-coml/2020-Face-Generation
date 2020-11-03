from tensorflow.keras.utils import Sequence
import numpy as np
import cv2

class ImageDataGen(Sequence):
    def __init__(self, ds, genModel, batchSize=64, imageSize=(160,192)):
        # Store basic values
        self.ds = ds
        self.model = genModel
        self.halfBatch = int(batchSize / 2)
        self.imageSize = imageSize
    
    def __getitem__(self, idx):
        # Get half a batch of real image data
        data = self.ds.skip(self.halfBatch * idx).take(self.halfBatch)
        realImages = [cv2.resize(np.array(element["image"]), dsize=self.imageSize, interpolation=cv2.INTER_CUBIC)/255.0 for element in data]
        
        # Get half a batch of fake image data
        fakeAttrs = [np.concatenate((np.random.randint(2, size=40), np.random.uniform(size=1))) for i in range(self.halfBatch)]
        fakeImages = np.reshape([self.model.predict(np.reshape(attr,(1,41))) for attr in fakeAttrs],(self.halfBatch, self.imageSize[1], self.imageSize[0], 3))
        
        # Combine halfs
        images = np.concatenate((realImages, fakeImages))
        truth = np.concatenate((np.ones((self.halfBatch,1), dtype=int), np.zeros(((self.halfBatch),1), dtype=int)))
        
        return images, truth

class ImageSequenceGenerator(Sequence):
    def __init__(self, dataset, size, model, image_size=(160, 192)):
        self.data = dataset
        self.batch_size = int(size / 2)
        self.model = model
        self.image_size = image_size

    def __getitem__(self, idx):
        images = self.data.skip(self.batch_size * idx).take(self.batch_size)
        real_image = [cv2.resize(np.array(i["image"]), dsize=self.image_size, interpolation=cv2.INTER_CUBIC) / 255.0 for
                      i in images]
        x_input = np.random.randint(2, size=(self.batch_size, 40))
        x_input = np.concatenate((x_input, np.random.uniform(size=(self.batch_size, 1))), axis=1)
        fake_image = np.reshape(self.model.predict(x_input),
                                (self.batch_size, self.image_size[1], self.image_size[0], 3))

        batch_x = np.concatenate((real_image, fake_image))
        batch_y = np.concatenate(
            (np.ones((self.batch_size, 1), dtype=int), np.zeros(((self.batch_size), 1), dtype=int)))

        return np.array(batch_x), np.array(batch_y)
    
class FakeAttributeDataGen(Sequence):
    def __init__(self, batchSize=64):
        # Store values
        self.batchSize = batchSize
        
    def __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        # Create fake attributes
        # attrs = np.array([np.concatenate((np.random.randint(2, size=40), np.random.uniform(size=1))) for i in range(self.batchSize)])
        attrs = np.random.normal(0, 1, (self.batchSize, 41))
        # Create truth array
        truth = np.ones(((self.batchSize),1), dtype=int)
        
        return attrs, truth
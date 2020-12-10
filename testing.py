import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import numpy as np
import cv2

from models import defineGenerator
from dataGens import ImageSequenceGenerator, FakeAttributeSequenceGenerator
from dataGens import dcgan


class FakeImageGenerator(Sequence):
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

        # fake_image = np.reshape(self.model.predict_on_batch(x_input),(self.batch_size,self.image_size[1],self.image_size[0],3))
        fake_image = np.reshape([self.model.predict(np.reshape(x, (1, 41))) for x in x_input],
                                (self.batch_size, self.image_size[1], self.image_size[0], 3))

        batch_x = np.concatenate((real_image, fake_image))
        batch_y = np.concatenate(
            (np.ones((self.batch_size, 1), dtype=int), np.zeros(((self.batch_size), 1), dtype=int)))

        return np.array(batch_x), np.array(batch_y)


    ds = tfds.load('celeb_a', data_dir="./", download=False)

    genModel = defineGenerator(41)

    dataGen = FakeImageGenerator(ds["test"], 4, genModel)
    print(dataGen[0][0][2])

    latentDim = 41
    genaModel = defineGenerator2(latentDim, 64, 64)
    discModel = defineDiscriminator2(64, 64)
    ganModel = defineGan2(genaModel, discModel)

    imageData = ImageDataGen(ds["test"], genaModel, imageSize=(64, 64))
    # testImageData = ImageSequenceGenerator(ds["test"], 32, genaModel, image_size=(64,64))
    attrData = FakeAttributeDataGen()

    train(genaModel, discModel, ganModel, ds["train"], batchSize=64, numBatch=20, numEpochs=2)
    testvals = attrData[0]
    ganModel.test_on_batch(testvals[0], testvals[1])
    predictions = genaModel.predict(attrData[0][0])
    genPred = genaModel.predict(attrData[0][0])

    print(ganModel.layers[2].predict(genPred))
    print(discModel.predict(genPred))

    print(ganModel.evaluate(attrData[0][0], attrData[0][1], verbose=0))
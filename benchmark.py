import requests
import time
import tarfile
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_datasets as tfds
from models import defineGenerator
from image_sequence_gen import ImageSequenceGenerator
from fake_attribute_sequence_generator import FakeAttributeSequenceGenerator

ds= tfds.load('celeb_a', data_dir=".", download=False)
ds_examples = ds["test"].take(16)

modelGen = defineGenerator(41, 64)

# import generator
modelGen.compile(loss='mse',
                  optimizer=Adam(1e-4, beta_1=0.5, decay=0.00005),
                  metrics=['mse'])

dataGen = ImageSequenceGenerator(ds["train"], 32, modelGen)

start_time = time.time()
for x in range(2):
    for y in range(2):
        print(dataGen[x][y].shape)
end_time = time.time()

print("time:", end_time - start_time)







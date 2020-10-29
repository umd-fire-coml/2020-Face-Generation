import requests
import tarfile
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from models import defineGenerator
from image_sequence_gen import ImageSequenceGenerator
from fake_attribute_sequence_generator import FakeAttributeSequenceGenerator

ds= tfds.load('celeb_a', data_dir=".", download=False)
ds_examples = ds["test"].take(16)

modelGen = defineGenerator(41, 64)

# import generator
dataGen = ImageSequenceGenerator(ds["train"], 32, modelGen)

#The batch of images
print(dataGen[10][0].shape == [32,192,160,3])

# The batch of 1s and 0s describing if the corresponding image is real or fake
print(dataGen[10][1].shape == [32,1])

fakeData = FakeAttributeSequenceGenerator()

#print shape of saachi's part
print(fakeData[0][1].shape)

#Print example of both parts[0][0][0] [0][1][0]
print(fakeData[0][0][0])
print(fakeData[0][1][0])

#print 2 or 3 images from sean's code
print(dataGen[0][0])
print(dataGen[0][1])






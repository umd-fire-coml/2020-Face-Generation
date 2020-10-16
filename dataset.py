import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.list_builders()[:20]

#change data_dir to where you want dataset to be stored
ds = tfds.load('celeb_a', data_dir=".")
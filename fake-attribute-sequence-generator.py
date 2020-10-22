import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random

class FakeAttributeSequenceGenerator(keras.utils.Sequence):
    
    def __init__(self, batch_size = 64):
        self.batch_size = batch_size
    
    def __getitem__():
        
        list0 = np.random.randint(2, size=(self.batchsize,40))
        row_add = np.random.uniform(size=(self.batchsize,1))
        list1 = np.concatenate((list0,row_add), axis=1)
        
        list2 = np.zeros((self.batchsize,1), dtype = int) 
        
        return list1, list2

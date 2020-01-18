import numpy as np
import keras
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './Preprocessing')
from CNNpreprocessing import *

class imgDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataPath, batch_size=1, shuffle=True):
        'Initialization'
        self.dataPath = dataPath
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        X, y = self.__data_generation(self.indexes[index])
        lenOfData=len(X[0])
        yL=np.zeros((lenOfData,len(y[0])))
        yL[:,:]=y
        return X, y#np.array([yL])

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        reload &update Datapath
        """
        #dataCohort: #groupsize,#videolength, pathToData
        self.dataCohort=prepdata(self.dataPath)
        self.labels,self.data=zip(*self.dataCohort)
        self.indexes = np.arange(len(self.dataCohort))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ID):
        'Generates data containing batch_size samples'
        """
        Speciallized for this Deep learning task, ID should be a single number
        """
        loadedData=imageListLoader(self.data[ID])
        dataLabel=self.labels[ID]

        return np.array([loadedData]), np.array([dataLabel])
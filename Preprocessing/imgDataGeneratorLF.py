import numpy as np
import keras
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './Preprocessing')
from preprocessingLF import *
from keras.preprocessing.image import ImageDataGenerator

class imgDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataPath, batch_size=32, shuffle=True):
        'Initialization'
        self.dataPath = dataPath
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.datagen = ImageDataGenerator(zca_whitening=True,rotation_range=90, width_shift_range=0.2, height_shift_range=0.2, brightness_range=[0.5,1.0], zoom_range=[0.85, 1.15], fill_mode='nearest', horizontal_flip=True, vertical_flip=True, data_format='channels_last')
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        XList=[]
        yList=[]
        # Generate data
        for i in range(self.batch_size):
            target=index*self.batch_size+i
            if target < len(self.indexes):
                X, y = self.__data_generation(self.indexes[index*self.batch_size+i])
#                 lenOfData=len(X[0])
#                 yL=np.zeros((lenOfData,len(y[0])))
#                 yL[:,:]=y
                XList.append(X)
                yList.append(y)
    
        return np.array(XList).astype('float32'), np.array(yList).astype('float32')

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
        randInt=random.randint(0,100)
        it1 = self.datagen.flow(np.array(loadedData[:1]), batch_size=1,seed = randInt)
        it2 = self.datagen.flow(np.array(loadedData[1:2]), batch_size=1,seed = randInt)
        it3 = self.datagen.flow(np.array(loadedData[2:]), batch_size=1,seed = randInt)
        d1 = it1.next()
        d2 = it2.next()
        d3 = it3.next()
        return np.array([d1[0],d2[0],d3[0]]), dataLabel
        #return np.array([loadedData]), np.array([dataLabel])
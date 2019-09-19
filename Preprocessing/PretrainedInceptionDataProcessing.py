#import numpy as np
import os
from PIL import Image
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from preprocessing import *
#import tensorflow as tf
#import keras
import plaidml.keras
plaidml.keras.install_backend()
#import keras
from keras import applications
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Reshape, Permute
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.merge import concatenate
#suppress warnings
import warnings
from functools import partial
import gc
import multiprocessing 
import threading
warnings.filterwarnings('ignore')
#from guppy import hpy

model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,input_shape = ( 244, 244, 3))
layer_name = 'mixed8'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

def intermediateOutput(dataInput):
    """
    dataInput with 4 dimension
    """
    model = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,input_shape = ( 244, 244, 3))
    layer_name = 'mixed8'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(dataInput)
    return intermediate_output
def saveOutput(path, name, file):
    np.save(os.path.join(path, name), file)

def checkOutput(path, name):
    if (os.path.exists(os.path.join(path, name+'.npy'))):
        return True
    else:
        return False
def makeDir(fullPath):
    if(not os.path.exists(fullPath)):
       os.makedirs(fullPath)
def loadData(file, invertData=False):
    """
    this function help prepare image data for input
    
    ## HUE SHIFT may be implemented here as well!!!
    
    file = file to be input, given as path
    invertData = if need to invertData
    """
    imgfd=Image.open(file, 'r')
    if(invertData):
        imgfd=imgfd.transpose(Image.FLIP_LEFT_RIGHT)
    img=np.array(imgfd)
    imgfd.close()
    img=(img/255-1)*2
    img=img.reshape(1,244,244,3)
    return img
#make function that is multiprocessable
def processImg(fileid, fileList, pathToList, invertData, destination):
    """
    Treat categories as multiprocessable resource
    
    fileList = list of categories
    fileid = #category to be processed
    pathToList = path to category
    invertData = if vertically invert the data
    destination = PARENT folder path to be saved to
    
    This function reads in image data, process the range to [-2,0], 
    and then input the data to inceptionV3 model
    
    then save the model as needed
    """
    
    categoryName=fileList[fileid]
    print(categoryName)
    fullpath=os.path.join(pathToList,categoryName)
    imgList=os.listdir(fullpath)
    if '.DS_Store' in imgList:
        imgList.remove('.DS_Store')
    #with tf.device('/GPU:0'):
    #ct=0
    skippedCount=0
    endSkip=False
    for img in imgList:
        #select 1 frame out of 5
        if(int(img[6:10])%5!=0):
            continue
        fileName=os.path.join(fullpath,img)
        destinationCategory=os.path.join(destination,categoryName)
        makeDir(destinationCategory)
        #allow for continuity after failures
        if (endSkip == False and checkOutput(destinationCategory,img[:-4])):
            #allow to continue work on unfinished data
            skippedCount+=1
            continue
        elif(endSkip == False):
            print("skipped, ", skippedCount,"images in ",categoryName)
            endSkip=True
        data=loadData(fileName,invertData)
        data=intermediateOutput(data)
        
        saveOutput(destinationCategory,img[:-4],data)
        gc.collect()
    
        

parentDataFolder='/home/livelab/Desktop/VideoData'
pathToDestination='/home/livelab/Desktop/Processed'
TTV=['Test','Validation','Training']
for folder in TTV:
    pathToFolder=os.path.join(parentDataFolder,folder)
    cateList=os.listdir(pathToFolder)
    if '.DS_Store' in cateList:
        cateList.remove('.DS_Store')
    #for category in cateList:
    if (folder == 'Test' and 'PULSE-OX' in cateList):
        cateList.remove('PULSE-OX')
    if (folder == 'Test' and  'ECG LEADS' in cateList):
        cateList.remove('ECG LEADS')
    #thCount=multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(int(thCount/2))
    partial_processImg=partial(processImg, fileList=cateList, pathToList=pathToFolder,
                                invertData=False, destination=pathToDestination)
    N = len(cateList)
    #with tf.device('/GPU:0'):
    #_=pool.map(partial_processImg,range(N))
    #pool.close()
    #pool.join()
    for i in range(N):
        partial_processImg(i)


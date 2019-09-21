"""
This file prepares data to be fed to CRNN model.
Prepare and seperate Data from data set
Labeling
"""
import os
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from functools import partial

def altPathJoin(path2, path1):
    return os.path.join(path1, path2)

def prepdata(path):
    labelList=[]
    dataList=[]
    #get one-hot label for categories
    labelLst=os.listdir(path)
    if '.DS_Store' in labelLst:
        labelLst.remove('.DS_Store')
    labelLst.sort()
    np_label=np.array(labelLst)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np_label)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    #setup counter to distinguish categories
    iter=0
    for category in labelLst:
        subDir=os.path.join(path, category)
        #currently in subject folders
        subjects=os.listdir(subDir)
        if '.DS_Store' in subjects:
            subjects.remove('.DS_Store')
        for subject in subjects:
            subsubDir=os.path.join(subDir, subject)
            clips=os.listdir(subsubDir)
            if '.DS_Store' in clips:
                clips.remove('.DS_Store')
            for clip in clips:
                #currently in clip folder, with all frame data
                frameData=np.array(os.listdir(os.path.join(subsubDir, clip)))
                data_length=len(frameData)
                if '.DS_Store' in frameData:
                    frameData.remove('.DS_Store')
                frameData.sort()
                ###############################################################
                #for data_length>200, randomly select a segment of 200 frames #
                ###############################################################
                if(data_length>200):
                    rangeOfData=data_length-200
                    startOfSegment=random.randint(0,rangeOfData-1)
                    frameData=frameData[startOfSegment:startOfSegment+200]
                pathToClip=altPathJoin(clip,subsubDir)
                partialAPJ=partial(altPathJoin, path1=pathToClip)
                fullPathFrameData=map(partialAPJ,frameData)
                labelList.append(onehot_encoded[iter])
                dataList.append(list(fullPathFrameData))
                #dataList.append(frameData)
        iter+=1
    return labelList, dataList


if __name__ == '__main__':
	testPath='/Users/lingfengli/Desktop/SIR/CRNN/ResizeBinnedToClips6F/Test'
	trainPath='/Users/lingfengli/Desktop/SIR/CRNN/ResizeBinnedToClips6F/Training'
	validPath='/Users/lingfengli/Desktop/SIR/CRNN/ResizeBinnedToClips6F/Validation'
	testl,testd=prepdata(testPath)
	trainl,traind=prepdata(trainPath)
	validl,validd=prepdata(validPath)

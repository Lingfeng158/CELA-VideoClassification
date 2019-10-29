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
from PIL import Image
import random

def makedir(dirName):
    cmd="mkdir {}"
    os.system(cmd.format( dirName))

def altPathJoin(path2, path1):
    return os.path.join(path1, path2)

def prepdata(path):
    """
    Can be run multiple times to get different data
    Used to generate data for majority vote
    return zipped [list_of_label, list_of_path_to_data]
    """
    framesToUse=3 # aim for 2 seconds of data
    framesGap=3 #4*5=20 frames between each selection #total uses 60 frames 
    categoryLength=400
    labelList=[]
    dataList=[]
    #get one-hot label for categories
    labelLst=os.listdir(path)
    if '.DS_Store' in labelLst:
        labelLst.remove('.DS_Store')
    labelLst.sort()
    #match up with keras
    labelLst=['ADMINISTER MEDICATION', 'BAGGING', 'BLOOD-PRESSURE CUFF', 'CHEST-TUBE', 'CHEST-TUBE PREP', 
              'COMBAT GAUZE', 'COMBAT TOURNIQUET', 'CPR (BREATH)', 'CPR (COMPRESSION)', 'DRAW MEDICATION', 
              'ECG LEADS', 'IM ADMINISTRATION', 'INTUBATION', 'IO LINE', 'IV LINE', 'IV TOURNIQUET', 
              'KING AIRWAY', 'ORAL AIRWAY', 'PULSE-OX', 'SPLINTING', 'SUTURING',
              'SWAB AREA WITH ALCOHOL', 'VITAL CHECKING', 'WRAP HEAD WOUND']
    np_label=np.array(labelLst)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np_label)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    #setup counter to distinguish categories
    iter=0
    for category in labelLst:
        categoryLable=[]
        categoryData=[]
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
                #############################################################
                #Select frames given specs until one clip runs out of frames#
                #############################################################
                tag=True
                while(tag):
                    #select a part of frames if data length is large
                    if(data_length>framesToUse*framesGap):
                        rd=random.randint(0,framesGap-1)
                        frameDataToUse=[frameData[i+rd] for i in np.array(range(framesToUse))*framesGap]
                        frameData=frameData[framesToUse*framesGap:]
                        data_length=len(frameData)
                        pathToClip=altPathJoin(clip,subsubDir)
                        partialAPJ=partial(altPathJoin, path1=pathToClip)
                        fullPathFrameData=map(partialAPJ,frameDataToUse)
                        categoryLable.append(onehot_encoded[iter])
                        categoryData.append(list(fullPathFrameData))
                    #elif not quiet enough
                    elif(data_length>(framesToUse-1)*framesGap):
                        frameDataToUse=[frameData[i] for i in np.array(range(framesToUse))*framesGap]
                        frameData=frameData[framesToUse*framesGap:]
                        data_length=len(frameData)
                        pathToClip=altPathJoin(clip,subsubDir)
                        partialAPJ=partial(altPathJoin, path1=pathToClip)
                        fullPathFrameData=map(partialAPJ,frameDataToUse)
                        categoryLable.append(onehot_encoded[iter])
                        categoryData.append(list(fullPathFrameData))
                    else:
                        tag=False
                    
                    #dataList.append(frameData)
        #print(len(categoryLable),category)
        if(len(categoryLable)<=categoryLength):
            labelList=labelList+categoryLable
            dataList=dataList+categoryData
        else:
            categoryCombo=list(zip(categoryLable,categoryData))
            random.shuffle(categoryCombo)
            categoryLable,categoryData = zip(*categoryCombo)
            labelList=labelList+list(categoryLable[:len(categoryLable)])
            dataList=dataList+list(categoryData[:len(categoryLable)])
        iter+=1
    return list(zip(labelList, dataList))

def imageListLoader(fileList):
    fileList.sort()
    #print(fileList)
    randInt=random.randint(0,1)
    invertData=False
    if (randInt==1):
        invertData=True
    resultList=[]
    for file in fileList:
        img=Image.open(file, 'r')
        if(invertData):
            img=img.transpose(Image.FLIP_LEFT_RIGHT)
        img=np.array(img)
        #img=(img/255-1)*2
        #img=img.reshape(1,244,244,3)
        resultList.append(img)
    return resultList

def npyListLoader(fileList):
    fileList.sort()
    resultList=[]
    data=np.load(filename)
    resultList.append(data)
    return resultList

def checkAndGeneratePath(parentPath):
    """
    This function takes in parent path for data, normally three folder should be under the folder:
    Test, Training and Validation (TTV)
    The function will check all subfolders have same subsubfolders,
    and return [subfolderlist]
    """
    folderList=os.listdir(parentPath)
    if '.DS_Store' in folderList:
        folderList.remove('.DS_Store')
    listOfcate=genList(parentPath,folderList)
    #snow ball canonical to include all categories appeared in elements of folderLst
    canonical=[]
    for folder in listOfcate:
        canonical=list(set(canonical).union(set(folder)))
    for folder in folderList:
        """
        folder = each TTV
        """
        pathToFolder=altPathJoin(folder,parentPath)
        itemList=os.listdir(pathToFolder)
        if '.DS_Store' in itemList:
            itemList.remove('.DS_Store')
        #figure out the lacking categories
        diff=list(set(canonical)-set(itemList))
        for category in diff:
            makedir(altPathJoin('"'+category+'"',pathToFolder))
    #at this point, all folder length should be equal
    if(checkEqual(parentPath,folderList)):
        print("Length Check on Folder Succeed")
    else:
        print("Error in TTV Folder Length")
        
def checkEqual(path, listOfFolder):
    """
    Return the position of element with largest value, and if 
    """
    listOfLength=[]
    for folder in listOfFolder:
        pathToFolder=altPathJoin(folder,path)
        itemList=os.listdir(pathToFolder)
        if '.DS_Store' in itemList:
            itemList.remove('.DS_Store')
        listOfLength.append(len(itemList))
    maxVal=max(listOfLength)
    sumVal=sum(listOfLength)
    if (sumVal==maxVal*len(listOfLength)):
        return True
    else:
        print('sum: ', sumVal, 'maxVal:', maxVal)
        return False

def genList(path, listOfFolder):
    """
    Return a list of lists of categories of elements in listOfFolder
    """
    listOfCate=[]
    for folder in listOfFolder:
        pathToFolder=altPathJoin(folder,path)
        itemList=os.listdir(pathToFolder)
        if '.DS_Store' in itemList:
            itemList.remove('.DS_Store')
        listOfCate.append(itemList)
    return listOfCate


if __name__ == '__main__':
	testPath='/Users/lingfengli/Desktop/SIR/CRNN/ResizeBinnedToClips6F/Test'
	trainPath='/Users/lingfengli/Desktop/SIR/CRNN/ResizeBinnedToClips6F/Training'
	validPath='/Users/lingfengli/Desktop/SIR/CRNN/ResizeBinnedToClips6F/Validation'
	testl,testd=prepdata(testPath)
	trainl,traind=prepdata(trainPath)
	validl,validd=prepdata(validPath)

## Filename: Explanation
ImageDataBinAug: classify data according to subject, clip, and perform any necessary data augmentation for lesser categories, then output organized softlink under specified destination  
jpgRange: sketch file to test ImageDataBinAug functions and find maximum values of each pixel  
Preprocessing: accessor function for ML algorithms to retrieve data from data structures, also responsible to pad any TTV folder to have same number of categories (by creating empty folders)  
ResizeImage: sketch file to test resizer functions  
Resizer: trim and resize images from original aspect ratio and size to asp=~1, size=256*256  
DataGenerator: generator function for Keras  

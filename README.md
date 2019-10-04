# CELA-VideoClassification
Classify and identify emergency procedurals performed in videos recorded in ambulance 

## Video Classification Using CRNN

The project is set to classify a group of already well splitted videos according to medical procedures being performed in the video.  

There are several challenges of this project:  
* Length of videos vary wildly, from 13 images to 5k images per session
* Each frame of the video is very similar. Traditional frame-based CNN method doesn't work.  

Two solutions are devised based on speculation:  
* Truncate any session with length larger than 500
* Train CNN and RNN in the same time

## Structure of Files:
Paper: all papers used for the project  
VideoDataProcessing: all files needed for processing video data and convert to frame data. Subject 1-5 are used for training, subject 6 is used for validation and subject 7 is used for testing. After processing, under the parent folder should be training, validation and test folder, under each of which should contains categories and all frames associate with each category.  
Preprocessing: all files needed for further turn framedata into ML ready data structure. All frame data will be organized according to : (TTV, category,) subject, clips

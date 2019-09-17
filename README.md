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
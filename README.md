# MonReader  
## Description  
Our company develops innovative Artificial Intelligence and Computer Vision solutions that revolutionize industries. Machines that can see: We pack our solutions in small yet intelligent devices that can be easily integrated to your existing data flow. Computer vision for everyone: Our devices can recognize faces, estimate age and gender, classify clothing types and colors, identify everyday objects and detect motion. Technical consultancy: We help you identify use cases of artificial intelligence and computer vision in your industry. Artificial intelligence is the technology of today, not the future.

MonReader is a new mobile document digitization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.  
  
![project_10_1](https://user-images.githubusercontent.com/98874284/222156736-d899d901-43a4-4d00-bb5b-9f427ad84511.png)
  
 MonReader is a new mobile document digitalization experience for the blind, for researchers and for everyone else in need for fully automatic, highly fast and high-quality document scanning in bulk. It is composed of a mobile app and all the user needs to do is flip pages and everything is handled by MonReader: it detects page flips from low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly, and it dewarps the cropped document to obtain a bird's eye view, sharpens the contrast between the text and the background and finally recognizes the text with formatting kept intact, being further corrected by MonReader's ML powered redactor.  
 
## Project Structure  
### Processing.py  
The Processing.py file contains the ImageProcessor class that preprocesses the input images before feeding them to the CNN model. The process method of this class takes the path to the folder containing the images and returns a list of processed images. This class used in oth Train and Predict class.  

### Train.py  
The Train.py file contains the CNNModel class that defines the architecture of the CNN model and trains it. The train method of this class takes the training and validation data, the number of epochs, and the batch size as inputs and trains the model. then you can run it and save the model to use it for predictions.  
### Predictions.py  
The Prediction.py file contains the Prediction class that predicts the flipping action in a sequence of images. The predict method of this class takes the path to the folder containing the test images and predicts the flipping action of each image. The pred_sequence method takes the paths to the folders containing the sequences of images and predicts if there is a flipping action in any of the sequences. This file y deafult will use pretrained model that allready trained and you can downlaod it from here **https://www.mediafire.com/file/i42t6l3emmn9v26/model_CNN.h5/file**  

## usage

Prepare the training and validation data by creating two folders named 0 and 1. The 0 folder should contain the images of not flipping, while the 1 folder should contain the images of flipping.  
Train the model by running the Train.py file.    
Predict the flipping action of a single image by running the Prediction.py file and entering the path to the folder containing the image.  
Predict the flipping action of a sequence of images by running the Prediction.py file and entering the paths to the folders containing the sequence of images.


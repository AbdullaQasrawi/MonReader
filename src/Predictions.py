#!/usr/bin/env python
# coding: utf-8

## This class contain two functions first one (predict) used to predict image label,and the second function (pred_sequence) used to predit if sequence of images have the action of fliping

from sklearn.metrics import classification_report
import logging
from Processing import ImageProcessor
from Train import CNNModel, run_training
from tensorflow.keras.models import load_model
import numpy as np

class Prediction:
    def __init__(self):
        self.processor = ImageProcessor()
        self.model = load_model("../models/model_CNN.h5")
        self.notflip = input("path to not flip images  ")
        self.flip = input("path to flip images  ")
        
        if not self.notflip or not self.flip:
            raise ValueError("Paths to images are required.")

        X_test_noflip = np.array(self.processor.process(self.notflip))
        X_test_flip = np.array(self.processor.process(self.flip))

        y_test_noflip = np.zeros(X_test_noflip.shape[0])
        y_test_flip = np.ones(X_test_flip.shape[0])
        
        if y_test_noflip.any() or X_test_flip.any():
            self.X_test = np.concatenate((X_test_noflip,X_test_flip ))
            self.y_test = np.concatenate((y_test_noflip, y_test_flip))
        elif y_test_noflip.any():
            self.X_test= X_test_flip
            self.y_test = y_test_flip
        else:
            self.X_test= X_test_noflip
            self.y_test = y_test_noflip
            

    def predict(self, threshold=0.5):
        test_pred = self.model.predict(self.X_test)
        test_pred_labels = np.where(test_pred >= threshold, 1, 0)
        return test_pred_labels
        
    def pred_sequence(self, threshold= 0.5):
        test_pred = self.model.predict(self.X_test)
        test_pred_labels = np.where(test_pred >= threshold, 1, 0)
        flip =1
        if flip in test_pred_labels:
            print("There is Fliping actions")
            logging.info("There is Fliping actions")
        else:
            print("There is no Fliping actions")
            logging.info("There is no Fliping actions")
if __name__ == '__main__':
    predictor = Prediction()
    predictor.predict()
    
    
    
    def predict(self, threshold=0.5):
        test_pred = self.model.predict(self.X_test)
        test_pred_labels = np.where(test_pred >= threshold, 1, 0)
        logging.info(classification_report(self.y_test, test_pred_labels))
        print(classification_report(self.y_test, test_pred_labels))
        logging.info(classification_report(self.y_test, test_pred_labels))
        
#!/usr/bin/env python
# coding: utf-8

import cv2
import os
from tqdm import tqdm


class ImageProcessor:
    def __init__(self, size=(70, 140)):
        self.size = size
        self.crop_coords = (0, 100, 0, 70)

    def process(self, folder):
        images = []
        for filename in tqdm(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ##convert images from BGR to RGB
            if img is not None:
                img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
                y, h, x, w = self.crop_coords
                img = img[y:y+h, x:x+w]
                img = img/255
                images.append(img)
                
        

        return images
        
        
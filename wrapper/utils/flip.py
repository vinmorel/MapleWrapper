# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:46:12 2020

@author: Vincent Morel
"""

import cv2
from os.path import join, isfile
from os import listdir

assets_pth = "C:/Users/vin_m/Desktop/BitBucket/MB/maplebot/testing/assets/"
partial_mobs_t = [cv2.imread(join(assets_pth, "monsters/test/", f),1) for f in listdir(join(assets_pth,"monsters/test/")) if isfile(join(assets_pth,"monsters/test/", f))]

for i, template in enumerate(partial_mobs_t):
    img = cv2.flip(template, 1)
    cv2.imwrite(f"{i}.png", img)
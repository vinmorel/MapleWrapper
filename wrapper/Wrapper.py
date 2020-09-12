# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:55:00 2020

@author: Vincent Morel
"""
import sys
import re
import pytesseract
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile
import d3dshot
from utils.window_pos import process_coords
from utils.nms import non_max_suppression_fast
import time
from PIL import Image, ImageDraw

class MapleWrapper():
    def __init__(self):
        self.p_coords = process_coords("MapleStory")
        self.p_w = self.p_coords[2] - self.p_coords[0]
        self.p_h = self.p_coords[3] - self.p_coords[1]
        self.gold = (self.p_w, self.p_w)
        self.assets_pth = "C:/Users/vin_m/Desktop/BitBucket/MB/maplebot/testing/assets/"
        self.name_t = cv2.imread(join(self.assets_pth,'NameTag2.png'),0)
        self.mobs_t = [cv2.imread(join(self.assets_pth, "monsters/", f),0) for f in sorted(listdir(join(self.assets_pth,"monsters/"))) if isfile(join(self.assets_pth,"monsters/", f))]
        # self.masks_t = [cv2.imread(join(self.assets_pth, "masks/", f),0) for f in sorted(listdir(join(self.assets_pth,"masks/"))) if isfile(join(self.assets_pth,"masks/", f))]
        self.slash_t = cv2.imread(join(self.assets_pth,'slash2.png'),0)
        self.bracket_t = cv2.imread(join(self.assets_pth,'bracket2.png'),0)

    def single_template_matching(self, img, template, method=cv2.TM_CCORR_NORMED):
        """
        returns int32 numpy array of best template match 
        [x0, y1, x1, y1]

        """
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        w, h = template.shape[::-1]
        
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        return np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], dtype=np.int32)

    def multi_template_matching(self, img, template, threshold=0.9, method=cv2.TM_CCOEFF_NORMED, nms=True):
        """
        returns int32 numpy array of all NMS filtered template matches above threshold 
        [x0, y1, x1, y1]

        """
        res = cv2.matchTemplate(img,template,method)
        loc = np.where( res >= threshold)
        
        w, h = template.shape[::-1]
        
        num_det = loc[0].shape[0]
        x2 = loc[1] + np.repeat(w,num_det) 
        y2 = loc[0] + np.repeat(h,num_det)
        boxes = np.column_stack((loc[1],loc[0], x2, y2))
        if nms:
            boxes = non_max_suppression_fast(boxes,0.2)
        return boxes

    def get_player(self):
        return self.single_template_matching(self.content, self.name_t)

    def get_mobs(self):
        entity_list = []
        for i, template in enumerate(self.mobs_t):
            entity_list += self.multi_template_matching(self.content, template, 0.6, cv2.TM_CCOEFF_NORMED, nms=False).tolist()
            
        entity_list = np.asarray(entity_list[:20], dtype=np.int32)
        
        return non_max_suppression_fast(entity_list, 0.6)

    def get_stats(self):
        stats = {
        'lvl' : (37, 594, 81, 622),
        'HP' : (243, 594, 0, 605),
        'MP' : (354, 594, 0, 605),
        'EXP' : (467, 594, 0, 605)
        }
        
        # self.multi_template_matching(self.ui, self.bracket_t)
        
        for stat in ['lvl', 'HP', 'MP', 'EXP']:
            pass

        return self.multi_template_matching(self.ui, self.bracket_t)

    def process_text(self, txt):
        txt = re.sub('[^0-9]','', txt)
        return int(txt)

    def ocr_text(self, img):
        ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        h,w = thresh.shape 
        thresh = cv2.resize(thresh,(w*3,h*3))
        
        ocr_result = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789./')
        
        return self.process_text(ocr_result)
    

    def start(self, fps=30):
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
        self.d.capture(target_fps=fps, region=self.p_coords)
        # warm up
        time.sleep(1)
        
        i = 0
        while True:  
            
            self.frame = cv2.cvtColor(self.d.get_latest_frame(), cv2.COLOR_BGR2GRAY)
            self.content =self.frame[0:int(0.882*self.p_h), :]
            self.ui = self.frame[int(0.9348 * self.p_h):, :int(0.7047 * self.p_w)]
            
            
            try:
                print(i)
                i += 1
                # player_box = self.get_player()
                
                print(self.get_player())
                print(self.get_mobs())
    
                # mob_boxes = self.get_mobs()
    
    
                # im = Image.fromarray(self.content)  
                # d = ImageDraw.Draw(im)
                # for box in mob_boxes:
                #     d.rectangle([(box[0],box[1]),(box[2],box[3])], outline ="red", width=6)
                
                # im.show()
                # ipt = input("press key to continue...")
                
            except (Exception) as e:
                print(e)
                self.d.stop()
                sys.exit()
        
            
    def stop(self):
        self.d.stop()

if __name__ == "__main__":
    from os.path import join 

    # d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
    w = MapleWrapper()
    print(cv2.TM_SQDIFF)
        # d.capture(target_fps=15, region=w.p_coords)
    # # warm up grace
    # time.sleep(0.1)
    w.start()
    
    # print(w.frame)
    
    # print(w.p_w)
    # print(w.p_h)
    # print(w.aspect_ratio)
    
    # w.start()
    # time.sleep(1)
    
    # print(w.frame)
    
    # print(w.get_stats())
    
    # d.stop()
    
    
    # print(w.multi_template_matching(cv2.imread(join(w.assets_pth,"1.png"),0), w.name_t))
    
    # print(w.ocr_text(cv2.imread(join(w.assets_pth,"161.png"),0)))
    # region_coords = process_coords("MapleStory")

    # d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
    # d.capture(target_fps=1, region=region_coords)
    
    # time.sleep(2)
    # d.stop()
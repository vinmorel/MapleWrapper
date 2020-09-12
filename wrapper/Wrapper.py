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
        self.numbers_t = [cv2.imread(join(self.assets_pth, "numbers/", f),0) for f in sorted(listdir(join(self.assets_pth,"numbers/"))) if isfile(join(self.assets_pth,"numbers/", f))]
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

    def multi_template_matching(self, img, template, threshold=0.7, method=cv2.TM_CCOEFF_NORMED, nms=True):
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
        coords = {
        # 'lvl' : [41, 9, 78, 33],
        'HP' : [243, 9, None, 18],
        'MP' : [354, 9, None, 18],
        'EXP' : [467, 9, None, 18]
        }
        
        slashes = self.multi_template_matching(self.ui, self.slash_t, 0.75)
        x1s = np.sort(slashes[:,0])        
        
        brackets = self.multi_template_matching(self.ui, self.bracket_t, 0.9)
        x2s = np.sort(brackets[:,0])
        
        buffer = 2
        
        coords['HP'][2] = x1s[0] + buffer
        coords['MP'][2] = x1s[1] + buffer
        coords['EXP'][2] = x2s[2] + buffer
        
        stats = []
        
        for k,v in coords.items():
            crop = self.ui[v[1]:v[3], v[0]:v[2]]
            
            stat = self.get_numbers(crop)
            # txt = self.ocr_text(crop)
            # txt = self.process_text(txt)
            # stats.append(txt)
            stats.append(stat)
        return stats

    def process_text(self, txt):
        txt = re.sub('[^0-9]','', txt)
        return int(txt)

    def ocr_text(self, img):
        ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        h,w = thresh.shape 
        thresh = cv2.resize(thresh,(w*3,h*3))
        
        ocr_result = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789./')
        
        return ocr_result
    
    def get_numbers(self, crop):
        # im = Image.fromarray(crop)
        # im.show()
        
        numbers_list = []
        for i, template in enumerate(self.numbers_t):
            im = Image.fromarray(crop)
            # im.show()
                        
            matches = self.multi_template_matching(crop, template, 0.95, cv2.TM_CCOEFF_NORMED)
            
            if type(matches) != list:
                for match in matches:
                    numbers_list.append([int(match[0]),str(i)])
        
        numbers_list = sorted(numbers_list, key = lambda x: int(x[0]))  
        # print(numbers_list)
        
        stat = ""
        
        for num in numbers_list:
            stat += num[1]
        
        return int(stat)
        
        # im = Image.fromarray(crop)  
        # d = ImageDraw.Draw(im)
        # if numbers_list:
        #     for box in numbers_list:
        #         box = box[0].tolist()
        #         d.rectangle([(box[0][0],box[0][1]),(box[0][2],box[0][3])], outline ="red", width=6)
        # im.show()

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
            
            # self.d.screenshot_to_disk(region=self.p_coords)
            
            cv2.imwrite("1.png",self.ui)
            
            # try:
                # print(i)
                # i += 1

                # print(self.get_player())
                # print(self.get_mobs())
            print(self.get_stats())
    
                # mob_boxes = self.get_mobs()
    
    
                # im = Image.fromarray(self.content)  
                # d = ImageDraw.Draw(im)
                # for box in mob_boxes:
                #     d.rectangle([(box[0],box[1]),(box[2],box[3])], outline ="red", width=6)
                
                # im.show()
                # input('...')
            # except (Exception) as e:
            #     print(e)
            #     self.d.stop()
            #     sys.exit()
         
    def stop(self):
        self.d.stop()

if __name__ == "__main__":
    w = MapleWrapper()
    w.start()
    

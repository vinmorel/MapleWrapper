# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:55:00 2020

@author: Vincent Morel
"""
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile
import d3dshot
from utils.window_pos import process_coords
from utils.nms import non_max_suppression_fast
import time
from PIL import Image, ImageDraw
import sys 
import concurrent.futures

class MapleWrapper():
    def __init__(self):
        self.p_coords = process_coords("MapleStory")
        self.p_w = self.p_coords[2] - self.p_coords[0]
        self.p_h = self.p_coords[3] - self.p_coords[1]
        self.gold = (806, 629)
        self.content_frame = [int(0.3*self.p_h), int(0.75*self.p_h), int(0.1*self.p_w), int(0.9*self.p_w)]
        self.ui_frame = [int(0.9348 * self.p_h), None, None, int(0.7047 * self.p_w)]
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
        player = self.single_template_matching(self.content, self.name_t)
        return player

    def get_mobs(self):
        """
        Returns list of list of mobs position [[x0, y1, x1, y1], ...]
        Currently must update template assets manually corresponding to mobs in map
        """
        ents = np.array([], dtype=np.int32)
        for i, template in enumerate(self.mobs_t):
            entities = self.multi_template_matching(self.content, template, 0.6, cv2.TM_CCOEFF_NORMED, nms=False)
            ents = np.append(ents, entities)
              
        size = ents.shape[0]
        chunks = size // 4
        
        if chunks != 0:
            ents = ents.reshape(chunks,-1)
            
        entity_list = ents[:10]
        entity_list = non_max_suppression_fast(entity_list, 0.8)
        return entity_list
    
    def get_stats(self):
        """
        Returns [HP, MP, EXP]
        Crops the UI into close ups of stat numbers dynamically with template matchings of x1 extremities.
        Matches numbers with crops to deduct the digits (alternative to using Tesseract which is very slow) 
        """
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
            stats.append(stat)
        return stats
    
    def get_numbers(self, crop):
        """
        Returns INT of numbers present in crop  
        """
        numbers_list = []
        for i, template in enumerate(self.numbers_t):                        
            matches = self.multi_template_matching(crop, template, 0.95, cv2.TM_CCOEFF_NORMED, nms=False)

            if type(matches) != list:                
                for match in matches:
                    numbers_list.append([int(match[0]),str(i)])
        
        numbers_list = sorted(numbers_list, key = lambda x: int(x[0]))  
        
        stat = ""
        
        for num in numbers_list:
            stat += num[1]
        
        return int(stat)
        
    def different_ratio(self):
        return (self.p_w != self.gold[0] or self.p_h != self.gold[1])

    def update_region(self, fps):
        p_coords = process_coords("MapleStory")
        if self.p_coords != p_coords:
            self.p_coords = p_coords
            self.d.stop()
            self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
            self.d.capture(target_fps=fps, region=self.p_coords)
            time.sleep(0.2) 
        return 'updated'

    def start(self, fps=25):
        """
        Starts extracting information from environment, given fps recommendation (slows down if computer can't 
        handle it). Merge this with agent. 

        """
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
        self.d.capture(target_fps=fps, region=self.p_coords)
        # let D3dshot warm up
        time.sleep(0.2)        
        
        i = 0
        while True:   
            self.update_region(fps)
            
            self.frame = cv2.cvtColor(self.d.get_latest_frame(), cv2.COLOR_BGR2GRAY)
            
            if self.different_ratio():
                print(self.p_w, self.p_h)
                self.frame = cv2.resize(self.frame, (self.p_w, self.p_h))
                self.p_w, self.p_h = self.gold
            
            self.content =self.frame[self.content_frame[0]:self.content_frame[1], self.content_frame[2]:self.content_frame[3]]
            self.ui = self.frame[self.ui_frame[0]:, :self.ui_frame[3]]
            
            # self.d.screenshot_to_disk(region=self.p_coords)
            
            # cv2.imwrite("1.png",self.ui)
            
            # try:
            print(i)
            i += 1    
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                player = executor.submit(self.get_player)
                stats = executor.submit(self.get_stats)
                mobs = executor.submit(self.get_mobs)
                a = player.result()
                b = stats.result()
                c = mobs.result()
            
            
                # im = Image.fromarray(self.content)  
                # d = ImageDraw.Draw(im)
                # for box in mob_boxes:
                #     d.rectangle([(box[0],box[1]),(box[2],box[3])], outline ="red", width=6)
                    
                # im.show()
                
            # im = Image.fromarray(w.content)
            # im.show()
            # im = Image.fromarray(w.ui)
            # im.show()

                # im2 = Image.fromarray(w.ui)
        
                # im.show()
                # im2.show()
                 
                # input('...')
                # except (Exception) as e:
                #     print(e)
                #     self.d.stop()
                #     sys.exit()
            # except Exception as e:
            #     print(e)
            #     self.d.stop()
            #     sys.exit()
                   
         
    def stop(self):
        self.d.stop()

if __name__ == "__main__":
    w = MapleWrapper()

    w.start()


        # im = Image.fromarray(crop)  
        # d = ImageDraw.Draw(im)
        # if numbers_list:
        #     for box in numbers_list:
        #         box = box[0].tolist()
        #         d.rectangle([(box[0][0],box[0][1]),(box[0][2],box[0][3])], outline ="red", width=6)
        # im.show()
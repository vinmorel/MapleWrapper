# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:55:00 2020

@author: Vincent Morel
"""

import cv2
import time
import pathlib
import d3dshot
import numpy as np
from os import listdir
import concurrent.futures
from os.path import join, isfile
from utils.NameTagMaker import make_tag
from utils.window_pos import process_coords, get_cname
from utils.nms import non_max_suppression_fast
from PIL import Image

class MapleWrapper():
    def __init__(self, player_name):
        self.wdir = pathlib.Path(__file__).resolve().parents[0]
        self.assets_pth = join(self.wdir,"templates")
        self.cname = get_cname('MapleStory')
        self.p_coords = process_coords(self.cname)
        self.p_w = self.p_coords[2] - self.p_coords[0]
        self.p_h = self.p_coords[3] - self.p_coords[1]
        self.gold = (806, 629)
        self.content_frame = [int(0.45*self.p_h), int(0.85*self.p_h), 0, int(self.p_w)]
        self.ui_frame = [int(self.p_h - 41.01), None, None, int(0.7047 * self.gold[0])]
        # self.name_t = cv2.imread(join(self.assets_pth,"general",'nametag.png'),0)
        self.name_t = make_tag(player_name)
        self.mobs_t = [cv2.imread(join(self.assets_pth, "mobs", f),0) for f in sorted(listdir(join(self.assets_pth,"mobs"))) if isfile(join(self.assets_pth,"mobs", f))]
        self.lvl_numbers_t = [cv2.imread(join(self.assets_pth, "numbers_lvl", f),0) for f in sorted(listdir(join(self.assets_pth,"numbers_lvl"))) if isfile(join(self.assets_pth,"numbers_lvl/", f))]
        self.numbers_t = [cv2.imread(join(self.assets_pth, "numbers", f),0) for f in sorted(listdir(join(self.assets_pth,"numbers"))) if isfile(join(self.assets_pth,"numbers", f))]
        self.slash_t = cv2.imread(join(self.assets_pth,"general","slash.png"),0)
        self.bracket_t = cv2.imread(join(self.assets_pth,"general","bracket.png"),0)

    def single_template_matching(self, img, template, method=cv2.TM_CCOEFF):
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
        Currently must update template assets manually corresponding to mobs in map.
        Leverages multi-processing.
        """
        ents = np.array([], dtype=np.int32)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            granular_entities = [executor.submit(self.multi_template_matching, self.content, template, 0.6, cv2.TM_CCOEFF_NORMED, nms=False) for template in self.mobs_t]
            for ent in granular_entities:
                ents = np.append(ents, ent.result())
            
            size = ents.shape[0]
            chunks = size // 4
            
            if chunks != 0:
                ents = ents.reshape(chunks,-1)
                
            entity_list = ents[:10]
            entity_list = non_max_suppression_fast(entity_list, 0.8)
            return entity_list
    
    def get_stats(self, investigate=False):
        """
        Returns [LVL, HP, MP, EXP]
        Crops the UI into close ups of stat numbers dynamically with template matchings of x1 extremities.
        Matches numbers with crops to deduct the digits (alternative to using Tesseract which is very slow) 
        """
        coords = {
            'LVL' : [40, 14, 76, 28],
            'HP' : [243, 9, None, 18],
            'MP' : [354, 9, None, 18],
            'EXP' : [467, 9, None, 18]
        }   
        
        x1_hp_mp = self.get_pos_x0(self.ui, self.slash_t, 0.75)
        x1_exp = self.get_pos_x0(self.ui, self.bracket_t, 0.9)
        
        buffer = 2
        
        coords['HP'][2] = x1_hp_mp[0] + buffer
        coords['MP'][2] = x1_hp_mp[1] + buffer
        coords['EXP'][2] = x1_exp[2] + buffer
        
        stats = []

        if investigate:
            for k,v in coords.items():
                crop = [v[0], v[1], v[2], v[3]]
                stats.append(crop)

        else:
            for k,v in coords.items():
                crop = self.ui[v[1]:v[3], v[0]:v[2]]
                if k == 'LVL':
                    stat = self.get_numbers(crop, self.lvl_numbers_t)
                else:
                    stat = self.get_numbers(crop, self.numbers_t)
                stats.append(stat)

        return stats

    def get_numbers(self, crop, templates):
        """
        Returns INT of numbers present in crop  
        """
        numbers_list = []
        for i, template in enumerate(templates):                        
            matches = self.multi_template_matching(crop, template, 0.95, cv2.TM_CCOEFF_NORMED, nms=False)

            if type(matches) != list:                
                for match in matches:
                    numbers_list.append([int(match[0]),str(i)])
        
        numbers_list = sorted(numbers_list, key = lambda x: int(x[0]))  
        stat = ""
        for num in numbers_list:
            stat += num[1]
        return int(stat)
        
    def get_pos_x0(self, ui, template, thresh):
        """
        Returns list [x1, ...] of the position(s) x0(s) of templates in UI.
        """
        x1 = self.multi_template_matching(ui, template, thresh)
        x1 = np.sort(x1[:,0])
        return x1
    
    def different_ratio(self):
        """ Returns Bool [True/False] if current window size is same as gold size"""
        return (self.p_w != self.gold[0] or self.p_h != self.gold[1])

    def update_region(self, fps):
        p_coords = process_coords(self.cname)
        if self.p_coords != p_coords:
            self.p_coords = p_coords
            self.d.stop()
            self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
            self.d.capture(target_fps=fps, region=self.p_coords)
            time.sleep(0.2) 
        return 'updated'

    def start(self, fps=25):
        """
        Starts capturing frames from environment, given fps recommendation (slows down if 
        computer can't handle it). 

        """
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
        self.d.capture(target_fps=fps, region=self.p_coords)     
        time.sleep(0.2)

    def get_aoi(self, game_window, clr_mode):
        """
        Crops the areas of interest (frame, content, ui) from the game_window given a color mode 
        [cv2.COLOR_BGR2GRAY or cv2.COLOR_RGB2BGR].
        """
        frame = cv2.cvtColor(game_window, clr_mode)
        content = frame[self.content_frame[0]:self.content_frame[1], self.content_frame[2]:self.content_frame[3]]
        ui = frame[self.ui_frame[0]:, :self.ui_frame[3]]
        return frame, content, ui

    def observe(self,fps=25, v=0):
        """
        Extracts information from the latest frame in buffer by leveraging multi-processing. 
        This information will be used by the agent. 
        """
        self.update_region(fps)
        self.frame, self.content, self.ui = self.get_aoi(self.d.get_latest_frame(), cv2.COLOR_BGR2GRAY)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            t1 = executor.submit(self.get_player)
            t2 = executor.submit(self.get_stats)
            t3 = executor.submit(self.get_mobs)
            player = t1.result()
            stats = t2.result()
            mobs = t3.result()
            if v: print('\n',player,'\n',stats,'\n',mobs)
    
    def stop(self):
        self.d.stop()

    def inspect(self, view, save_to_disk=False):
        """
        Displays an image and template detections of a view for debugging. 
        views : [frame, content, ui, player, mobs, stats]
        """
        self.d = d3dshot.create(capture_output="numpy", frame_buffer_size=50)
        game_window = self.d.screenshot(region=self.p_coords)  
        
        clr_frame, clr_content, clr_ui = self.get_aoi(game_window, cv2.COLOR_RGB2BGR)
        self.frame, self.content, self.ui = self.get_aoi(game_window, cv2.COLOR_BGR2GRAY)

        items = {
            'frame' : [clr_frame, None],
            'content' : [clr_content, None],
            'ui' : [clr_ui, None],
            'player' : [clr_content, self.get_player],
            'mobs' : [clr_content, self.get_mobs],
            'stats' : [clr_ui, self.get_stats]
        }

        image = items[view][0]
        clr = (0, 0, 255)
        width = 1
  
        if items[view][1] != None:
            boxes = items[view][1]() if (view != 'stats') else items[view][1](True)
            if view == 'player':
                image = cv2.rectangle(image, (boxes[0],boxes[1]), (boxes[2],boxes[3]), clr, width) 
            else:
                for box in boxes:
                    image = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), clr , width)
        
        if save_to_disk:
            cv2.imwrite(f"{view}.png", image)

        self.d.stop()
        cv2.imshow(f"{view}", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":   
    w = MapleWrapper('smashy')
    w.start()
    
    i = 0
    while True:
        w.observe(v=1)
        i += 1

    # w.inspect('player')
        


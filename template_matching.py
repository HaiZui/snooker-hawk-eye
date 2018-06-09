# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:00:23 2018

@author: haizui
"""


import cv2
import numpy as np

def find_balls(img_gray):
    template_ball1 = cv2.imread('./templates/ball_red.png',0)
    template_ball2 = cv2.imread('./templates/ball_red2.png',0)
    template_ball3 = cv2.imread('./templates/ball_black.png',0)
    #template_ball2 = cv2.imread('ball2.png',0)
    template_ball4 = cv2.imread('./templates/cue.png',0)
    template_ball5 = cv2.imread('./templates/ball_red3.png',0)
    template_ball6 = cv2.imread('./templates/ball_red4.png',0)
    template_ball7 = cv2.imread('./templates/ball_brown.png',0)
    template_ball8 = cv2.imread('./templates/ball_red5.png',0)
    template_ball9 = cv2.imread('./templates/ball_red6.png',0)
    template_ball10 = cv2.imread('./templates/ball_red7.png',0)
    template_ball11 = cv2.imread('./templates/ball_green.png',0)
    
    templates = [template_ball1
                 , template_ball2
                 , template_ball3
                 , template_ball4
                 , template_ball5
                 , template_ball6
                 , template_ball7
                 , template_ball8
                 , template_ball9
                 , template_ball10
                 , template_ball11]
    thresholds = [0.88, 0.90, 0.88, 0.88, 0.86, 0.86, 0.86, 0.86, 0.86, 0.85, 0.85  ]

    results = []
    for i in range(len(templates)):    
        
        template = templates[i]
        threshold = thresholds[i]
        w, h = template.shape[::-1]  

        res_ball = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)        
        loc_ball = np.where( res_ball >= threshold)
        for pt in zip(*loc_ball[::-1]):
            results.append(tuple(i1+i2 for i1, i2 in zip(pt,(int(w/2),0))))
    
    return remove_duplicate_points(results)

def remove_duplicate_points(pts):
    # Pixels for threshold length = how far away the points need to be that  
    # they are interpret as two separate points    
    threshold_length = 5
    i = 0
    res = pts.copy()
    while i < len(res):
        for j in reversed(range(i+1,len(res))):
            x1 = res[i][0]
            y1 = res[i][1]
            x2 = res[j][0]
            y2 = res[j][1]
            if np.sqrt((x1-x2)**2+(y1-y2)**2) < threshold_length:
                res.pop(j)
        i += 1
    return res
    
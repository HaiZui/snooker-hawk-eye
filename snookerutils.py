# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:03:00 2018

@author: haizui
"""

import cv2
import numpy as np
from geometry import intersectLines, Line
import matplotlib.pyplot as plt

class Table:
    def __init__(self
                 , top
                 , right
                 , bottom
                 , left
                 ):
             self.top = top
             self.right = right
             self.bottom = bottom
             self.left = left
             
    def find_corners(self):
        left_top = intersectLines(self.top.pt1, self.top.pt2, self.left.pt1, self.left.pt2)
        left_bottom = intersectLines(self.bottom.pt1, self.bottom.pt2, self.left.pt1, self.left.pt2)
        right_top = intersectLines(self.top.pt1, self.top.pt2, self.right.pt1, self.right.pt2)
        right_bottom = intersectLines(self.bottom.pt1, self.bottom.pt2, self.right.pt1, self.right.pt2)
        return np.array([left_top, left_bottom, right_top, right_bottom])

def find_snooker_table_mask(img_rgb):
    # Lower and upper RGB values for finding green segments from the image
    lower = np.array([0,90,0])
    upper = np.array([90,185,90])
    # Only green parts from the image
    shapeMask = cv2.inRange(img_rgb, lower, upper)
    
    # Bit more processing to find out the whole shape
    kernel = np.ones((50, 50), np.uint8)
    closing = cv2.morphologyEx(shapeMask, cv2.MORPH_CLOSE, kernel)

    return closing
    
def find_green(img_rgb):
    # Lower and upper RGB values for finding green segments from the image
    lower = np.array([0,90,0])
    upper = np.array([90,185,90])
    # Only green parts from the image
    shapeMask = cv2.inRange(img_rgb, lower, upper)
    
    return shapeMask
    
def find_black(img_rgb):
    # Lower and upper RGB values for finding green segments from the image
    lower = np.array([0,0,0])
    upper = np.array([15,15,15])
    # Only green parts from the image
    shapeMask = cv2.inRange(img_rgb, lower, upper)
    
    # Bit more processing to find out the whole shape
    kernel = np.ones((30, 30), np.uint8)
    closing = cv2.morphologyEx(shapeMask, cv2.MORPH_CLOSE, kernel)
    
    return closing
    
def find_white(img_rgb):
    # Lower and upper RGB values for finding green segments from the image
    lower = np.array([200,200,150])
    upper = np.array([255,255,255])
    # Only green parts from the image
    shapeMask = cv2.inRange(img_rgb, lower, upper)
    
    # Bit more processing to find out the whole shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closing = cv2.morphologyEx(shapeMask, cv2.MORPH_CLOSE, kernel)
    
    return closing
    
def find_snooker_table_corners(img_rgb):
    len_y, len_x, len_z = np.shape(img_rgb)
    closing = find_snooker_table_mask(img_rgb)
    edges = cv2.Canny(closing, 100, 200)
    
    # Finding lines from the green mask edges
    minLineLength = 100
    maxLineGap = 100
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, minLineLength, maxLineGap)
    
    # Find out which edges of the table each line represents
    top_lines = np.array([])
    left_lines = np.array([])
    right_lines = np.array([])
    bottom_lines = np.array([])
    
    for line in lines:
        (x1,y1,x2,y2) = line[0]
        lineo = Line(x1, y1, x2, y2)
        # top
        if lineo.angle() < 30 and y1 < len_y/2:
            top_lines = np.append(top_lines, lineo)
        # bottom
        if lineo.angle() < 30 and y1 > len_y/2:
            bottom_lines = np.append(bottom_lines, lineo)
        # left
        if lineo.angle() > 30 and x1 < len_x/2:
            left_lines = np.append(left_lines, lineo)
        # right
        if lineo.angle() > 30 and x1 > len_x/2:
            right_lines = np.append(right_lines, lineo)
            
    # Find best = longest lines of each type
    # Top
    for line in top_lines:
        line_lengths = np.array([])
        line_lengths = np.append(line_lengths,[line.length()])
        top_line = top_lines[np.argmax(line_lengths)]
    # Bottom
    for line in bottom_lines:
        line_lengths = np.array([])
        line_lengths = np.append(line_lengths,[line.length()])
        bottom_line = bottom_lines[np.argmax(line_lengths)]
    # Left
    for line in left_lines:
        line_lengths = np.array([])
        line_lengths = np.append(line_lengths,[line.length()])
        left_line = left_lines[np.argmax(line_lengths)]
    # Right
    for line in right_lines:
        line_lengths = np.array([])
        line_lengths = np.append(line_lengths,[line.length()])
        right_line = right_lines[np.argmax(line_lengths)]
        
    # Only best lines
    lines = [top_line, bottom_line, left_line, right_line]
      
    # Create Table-object
    table = Table(top=top_line, bottom=bottom_line, left=left_line, right=right_line)
    corners = table.find_corners()
      
    return corners
   
def compare_rgb(rgb_ar):
    
    color_recognized = 'Table'
    if rgb_ar is None:
        return color_recognized

    rgb_ar = np.array(rgb_ar)
    colors = {
                'Table':
                np.array([1, 103, 0])
                ,
                'Cue':
                np.array([190, 246, 254])
                ,
                'Red':
                np.array([18, 40, 159])
                ,
                'Black':
                np.array([28, 31, 21])
                ,
                'Pink':
                np.array([160, 180, 240])
                ,
                'Blue':
                np.array([173, 105, 12])
                ,
                'Brown':
                np.array([52, 114, 155])
                ,
                'Green':
                np.array([77, 93, 4])
                ,
                'Yellow':
                np.array([48, 225, 254])
            }
        
    goodness = 255*3
    for color in colors.items():
        goodness_cur = np.sum(abs(color[1]-rgb_ar))
        #print(goodness_cur, goodness, color)
        if goodness_cur < goodness:
            color_recognized = color[0]
            goodness = goodness_cur
    return color_recognized
    
def select_ball(pt, img_rgb):
    # Radius of balls in pixels
    r_ball = 10
    pt1 = (pt[0]-r_ball, pt[1]+r_ball)
    pt2 = (pt[0]+r_ball, pt[1])

    width = np.shape(img_rgb)[0]
    height = np.shape(img_rgb)[1]
    mask = np.zeros((width,height), np.uint8)
    cv2.rectangle(mask, pt1, pt2, 255, thickness=-1)
    #
    #mask_green = find_green(img_rgb)
    ##invert
    #mask = cv2.bitwise_not(mask)
    
    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    # Remove table parts from the image
    mask_green = find_green(res)
    #invert
    mask_green = cv2.bitwise_not(mask_green)
    
    res = cv2.bitwise_and(res, res, mask=mask_green)

    return res
    
def average_rgb(img_rgb):
    # Store rgb values into one-dimensional array
    rgb_values = img_rgb.reshape((img_rgb.shape[0]*img_rgb.shape[1],img_rgb.shape[2]))
    # Find non-zero pixels (=pure black)
    non_zeros = (np.sum(rgb_values, axis=1) != 0)
    # Only non-black pixels considered
    rgb_values = rgb_values[non_zeros]
    # Return element-wise mean 
    ret = None
    if len(rgb_values) > 20:# Only arrays with enoug points will be considered
        ret = rgb_values.mean(axis=0)
    return ret

def identify_ball(pt, img_rgb):    
    res = select_ball(pt, img_rgb)
    average_color = average_rgb(res)
    
    return compare_rgb(average_color)

def identify_balls(pts, img_rgb):
    colors = {
            'Table':
            np.array([1, 103, 0])
            ,'Cue':
            np.array([190, 246, 254])
            ,'Red':
            np.array([18, 40, 180])
            ,'Black':
            np.array([28, 31, 21])
            ,'Pink':
            np.array([160, 180, 240])
            ,'Blue':
            np.array([173, 105, 12])
            ,'Brown':
            np.array([52, 114, 155])
            ,'Green':
            np.array([77, 130, 4])
            ,'Yellow':
            np.array([48, 225, 254])
            }
    
    unique_balls = ['Cue','Black','Pink','Blue','Brown','Green','Yellow']  
#    # Images of balls 
#    ball_imgs = [select_ball(pt, img_rgb) for pt in pts]
#    # Average RGB:s of found balls
#    ave_rgbs = [average_rgb(ball_img) for ball_img in ball_imgs]
    
    ave_rgbs = {tuple(pt):average_rgb(select_ball(pt, img_rgb)) for pt in pts}
    ave_rgbs = {key:val for key, val in ave_rgbs.items() if val is not None}
    results = {tuple(pt):'Table' for pt in pts}
    # Identify unique balls, which should have only one occurence
    for ball in unique_balls:
        goodnesses = {tuple(pt):goodness(ave_rgbs[pt], colors[ball]) for pt in ave_rgbs.keys()}
        best = min(goodnesses,key=goodnesses.get)
        results[best] = ball
        del ave_rgbs[best]
    # Identify red balls and remove 
    for ball in ave_rgbs.items():
        goodness_red = goodness(ball[1],colors['Red'])
        goodness_table = goodness(ball[1],colors['Table'])
        
        if goodness_red < goodness_table:
            results[ball[0]] = 'Red' 
    return results

def goodness(c1, c2):
    return np.sum(abs(c1-c2))
    
def unwarp(img, src, dst, testing):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(cv2.flip(warped, 1))
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
    else:
        return warped
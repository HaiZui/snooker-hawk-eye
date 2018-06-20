# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:16:00 2018

@author: haizui
"""





import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', help='foo help')
args = parser.parse_args()

path = args.path

def ball_colors(color):
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
            np.array([10, 10, 10])
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
    return colors[color]
    
if __name__ == "__main__":
    import cv2
    import numpy as np
    import snookerutils as snu
    import visualize       
    import matplotlib.pyplot as plt 
    import template_matching as tm
    # Font for printed labels        
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Some constants
    # Resized image size
    w_scaled = 1366.
    # Cushion width has to be compensated
    cushion = 7
    table_measure_cm = (382.+2*cushion,204.+2*cushion)
    ball_radius_cm = 52.5/2 * 10**-1
    
    # Start
    img_rgb = cv2.imread(path)

    h,w,_ = np.shape(img_rgb)

    # Resize image
    scale = w_scaled / w
    w_new = int(w*scale)
    h_new = int(h*scale)
    img_rgb = cv2.resize(img_rgb, (w_new, h_new))

    # Ball radius in pixels
    ball_radius = h*ball_radius_cm/max(table_measure_cm)

    # Find corner positions
    corners = snu.find_snooker_table_corners(img_rgb)
    
    # Note coordinates need to be converted to np.float32 for 
    # cv2.getPerspectiveTransform function
    corners_true = np.array([(0, 0),
                              (0, h),
                              (table_measure_cm[1]/table_measure_cm[0]*h, 0),
                              (table_measure_cm[1]/table_measure_cm[0]*h, h)], np.float32)
    
    # Perform perspective correction
    img_rgb_unwarped = snu.unwarp(img_rgb, np.array(corners, np.float32), corners_true, False)
    
    # Perspective correction matrix
    M = cv2.getPerspectiveTransform(np.array(corners,np.float32), corners_true)
    
    for corner in corners:
        print(corner)
        cv2.circle(img_rgb, (int(corner[0]), int(corner[1])), 10, (255,0,255) ) 

    mask = snu.find_snooker_table_mask(img_rgb)
    
    # Only table extracted from the original image
    img_table= cv2.bitwise_and(img_rgb,img_rgb,mask = mask)    
   
    img_gray = cv2.cvtColor(img_table,cv2.COLOR_BGR2GRAY)
    
    #==============================================================================
    #     Finding balls by template matching   
    #==============================================================================
    
    loc_balls = tm.find_balls(img_gray)
    # Identify colors
    identified = snu.identify_balls(np.array(loc_balls), img_rgb)
    
    # Cue ball location, which is used later to move origin
    cue_loc = list({k:v for k, v in identified.items() if v == 'Cue'}.keys())[0]
    
    # Perform perspective transformation to real table coordinates
    loc_balls_true = cv2.perspectiveTransform(np.array([np.array(loc_balls, dtype='float32')]), M)[0]
    # Store original and transformed coordinates into dict
    loc_balls_map = {loc_balls[i]:loc_balls_true[i] for i in range(len(loc_balls))}    
    
    cue_loc_true = loc_balls_map[tuple(cue_loc)]
    
    # Move origin to cue-ball
    loc_balls_true = loc_balls_true - np.array(cue_loc_true)
    corners_true = corners_true - np.array(cue_loc_true)
    
    
    print('True ball locations: ',loc_balls_true)
    print('Table corner locations: ',corners)
    print('True table corner locations: ',corners_true)
    print("Found {} balls".format(len(loc_balls)))
    # Draw ball locations 
    
    
    for ball in identified.items():
        color = ball[1]
        pt = ball[0]
        pt_true = loc_balls_map[pt]
        print("{} ball at ".format(color), pt)
        if color != 'Table':
            print('drawing ball at', pt_true-cue_loc_true)
            cv2.putText(img_rgb,color,pt, font, 0.5,(255,255,255),2,cv2.LINE_AA)
            cv2.circle(img_rgb,pt,2,(0,0,255),3)
            visualize.draw_ball(pt_true-cue_loc_true, ball_radius, tuple(ball_colors(color)[::-1]/255))
    visualize.draw_snooker_table(pt1=corners_true[0], pt2=corners_true[-1], z=-ball_radius)
    #plt.imshow(img_gray)
    plt.imshow(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB))
    plt.show()

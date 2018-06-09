# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:32:50 2018

@author: haizui
"""

import vpython as vp
from numpy import pi


scene = vp.canvas(title='Paha paikka',
     x=0, y=0, width=1280, height=768,
     center=vp.vector(0,0,0), background=vp.vector(0,0,0))

# Reset lightning
scene.lights = []
vp.distant_light(direction=vp.vec( 0.0,  1,  0.0), color=vp.vector(1,1,1))
vp.local_light(pos=vp.vec( 0.0,  700,  0.0), color=vp.vector(1,1,1))
## Create table
#table = vp.box(pos=vp.vec(0,-15,0)
#                , length = 1400
#                , height = 10
#                , width = 1400
#                , color=vp.vec(0,103/255,1/255)
#                , texture={'file':'./snooker_texture.jpg','place':'all'}
#                )
#table.shininess=10000


def draw_ball(pt, radius, color):
    vp.sphere(pos=vp.vector(pt[0],0,pt[1]), radius=radius, color=vp.vector(color[0],color[1],color[2]))
    

def draw_snooker_table(pt1, pt2, z):
    # pt1 = left top corner
    # pt2 = right bottom corner
    # z = ball radius
    print('Drawing table', pt1, pt2, z)
    length = pt2[0]-pt1[0]
    width = pt2[1]-pt1[1]
    height = 10
    # Create table, note that the position is determined by the center of the box
    table = vp.box(pos=vp.vec(pt2[0]-length/2,z-height/2,pt2[1]-width/2)
                    , length = int(length)
                    , height = 10
                    , width = int(width)
                    , color=vp.vec(0,103/255,1/255) # Green
                    , texture={'file':'./snooker_texture.jpg','place':'all'}
                    )
    table.shininess=100000
    
    corner_pocket_br = vp.extrusion(path=vp.paths.arc(radius=height,
                                                   angle1=-pi/2, angle2=+pi/2)
                                   , color=vp.vector(0.8,0.8,0.8)
                , shape=[[vp.shapes.rectangle(pos=(1,0),
              width=height,height=height)] ])
#              
    corner_pocket_br.rotate(angle=3*pi/4, axis=vp.vec(0,1,0))
    corner_pocket_br.pos = vp.vec(pt1[0], 0, pt1[1])
    corner_pocket_tr = corner_pocket_br.clone(pos=vp.vector(pt1[0], 0, pt2[1]))
    corner_pocket_tr.rotate(angle=pi/2, axis=vp.vec(0,1,0))
    corner_pocket_tl = corner_pocket_tr.clone(pos=vp.vector(pt2[0], 0, pt2[1]))
    corner_pocket_tl.rotate(angle=pi/2, axis=vp.vec(0,1,0))
    corner_pocket_bl = corner_pocket_tl.clone(pos=vp.vector(pt2[0], 0, pt1[1]))
    corner_pocket_bl.rotate(angle=pi/2, axis=vp.vec(0,1,0))
    
#    path = [ [0,0], [0,20], [10,50]]
#    vp.extrusion(path=path, shape=[[vp.shapes.rectangle(pos=(1,0),
#              width=height,height=height)]], color=vp.color.red)
    

if __name__ == '__main__':
    pt1 = (-155.65356445, -159.0714111) 
    pt2 = ( 267.13430786,  608.92858887)
    z = -5.090909090909091
    draw_snooker_table(pt1,pt2,z)
    scene.center()
#    L = [(-1,0,-0.5), (-1,0,0.5)]
#    A = vp.paths.arc(pos=vp.vec(1,0,0), radius=0.5,
#                angle1=-0.5*pi, angle2=0.5*pi)
#    p = vp.vec(0,0,0)
#    print(p)
#    vp.extrusion(pos=p, shape=vp.shapes.triangle(length=0.4))
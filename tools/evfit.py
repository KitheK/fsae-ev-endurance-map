import math, numpy as np
from PIL import Image
from geo import map2z18, RES18
MPP_EV=30*0.3048/17.95
def frame(u0z,v0z,bear,mpp,size,flip=False):
    # returns affine coeffs: frame px (u,v) -> z18 px, origin frame(0,0) at z18 (u0z,v0z); u along bearing, v toward bearing+90 (right side) 
    b=math.radians(bear); k=mpp/RES18
    rx,ry=math.sin(b),-math.cos(b)          # along
    dx,dy=math.cos(b),math.sin(b)            # down = bearing+90
    if flip: rx,ry,dx,dy=-rx,-ry,-dx,-dy
    return (k*rx,k*dx,u0z,k*ry,k*dy,v0z)
def render(coef,size,out):
    sat=Image.open('mis_z18.jpg').convert('RGB')
    sat.transform(size,Image.AFFINE,coef,resample=Image.BILINEAR).save(out)

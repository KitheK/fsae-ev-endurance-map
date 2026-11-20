import math, sys
from PIL import Image
import numpy as np
from ov import MPP_MAP, RES
def sat2map(ax,ay,bearing,scale=1.0,out='/tmp/s2m.jpg',blend=True):
    sat=Image.open('mis_z18.jpg').convert('RGB')
    b=math.radians(bearing); k=MPP_MAP*scale/RES
    rx,ry=math.sin(b),-math.cos(b); ux,uy=-math.cos(b),-math.sin(b)
    # map(u,v)->sat: x=ax+k*((u-1383)*rx - (v-522.5)*ux)
    A=k*rx; B=-k*ux; C=ax-k*(1383*rx-522.5*ux)
    D=k*ry; E=-k*uy; F=ay-k*(1383*ry-522.5*uy)
    w=sat.transform((1600,693),Image.AFFINE,(A,B,C,D,E,F),resample=Image.BILINEAR)
    if blend:
        mp=np.array(Image.open('endurance.png').convert('RGB')).astype(int)
        L=mp.mean(2); red=(mp[...,0]>200)&(mp[...,1]<150)
        W=np.array(w).copy()
        W[L<100]=[255,255,0]; W[red]=[255,40,40]
        w=Image.fromarray(W.astype(np.uint8))
    w.save(out)
if __name__=='__main__':
    a=list(map(float,sys.argv[1:5])); sat2map(*a)

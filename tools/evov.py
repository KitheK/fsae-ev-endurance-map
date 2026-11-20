import numpy as np, math, sys
from PIL import Image
from evfit import *
from geo import map2z18
cx,cy=map2z18(810,580)
def evcoef(du,dv,s=1.0,bear=13.5):
    c=frame(0,0,bear,MPP_EV,(1700,400),True); A,B,_,D,E,_=c
    C=cx-(A*850+B*200); F=cy-(D*850+E*200)
    # frame=(ev*s+off) -> z18
    return (A*s,B*s,A*du+B*dv+C, D*s,E*s,D*du+E*dv+F)
def ev2z18(u,v,co):
    A,B,C,D,E,F=co; return A*np.asarray(u)+B*np.asarray(v)+C, D*np.asarray(u)+E*np.asarray(v)+F
def show(du,dv,s=1.0,bear=13.5,out='/tmp/evov.png'):
    co=evcoef(du,dv,s,bear)
    sat=Image.open('mis_z18.jpg').convert('RGB').transform((1600,191),Image.AFFINE,co,resample=Image.BILINEAR)
    mp=np.array(Image.open('ev_official.png').convert('RGB')).astype(int); L=mp.mean(2)
    red=(mp[...,0]>190)&(mp[...,0]-mp[...,1]>50)
    W=np.array(sat).copy(); W[(L<90)&~red]=[255,255,0]; W[red]=[255,30,30]
    Image.fromarray(W.astype(np.uint8)).resize((3200,382)).save(out)
if __name__=='__main__':
    show(*map(float,sys.argv[1:]))

import numpy as np, math, sys
from PIL import Image
MPP_MAP=0.3048*100/61.9   # m per map px
RES=0.4437                # m per z18 px (approx at lat 42.067)
def make(ax,ay,bearing,scale=1.0,out='/tmp/ov.jpg',crop=None,alpha=1.0):
    sat=Image.open('mis_z18.jpg').convert('RGBA')
    mp=np.array(Image.open('endurance.png').convert('RGBA')).astype(float)
    # keep only non-white
    L=mp[...,:3].mean(2)
    red=(mp[...,0]>200)&(mp[...,1]<150)
    a=np.where(red|(L<100),255,0).astype(float)
    rgb=mp[...,:3].copy(); rgb[~red]=[255,255,0]  # lines yellow
    m=np.dstack([rgb,a*alpha]).astype(np.uint8)
    mi=Image.fromarray(m)
    from PIL import ImageFilter
    al=mi.getchannel('A').filter(ImageFilter.MaxFilter(3)); mi.putalpha(al)
    b=math.radians(bearing)
    k=MPP_MAP*scale/RES
    # map px (u,v) -> sat px: offset from origin (1383,522.5)
    rx,ry=math.sin(b),-math.cos(b)          # map right
    ux,uy=-math.cos(b),-math.sin(b)          # map up (bearing-90)
    # affine sat->map for PIL: u = 1383 + ((x-ax)*rx+(y-ay)*ry)/k ; v = 522.5 - ((x-ax)*ux+(y-ay)*uy)/k
    A=rx/k;B=ry/k;C=1383-(ax*rx+ay*ry)/k
    D=-ux/k;E=-uy/k;F=522.5+(ax*ux+ay*uy)/k
    w=mi.transform(sat.size,Image.AFFINE,(A,B,C,D,E,F),resample=Image.BILINEAR)
    o=Image.alpha_composite(sat,w).convert('RGB')
    if crop: o=o.crop(crop)
    o.save(out); return (A,B,C,D,E,F)
if __name__=='__main__':
    a=list(map(float,sys.argv[1:5])); make(*a,out=sys.argv[5] if len(sys.argv)>5 else '/tmp/ov.jpg')

import numpy as np, pickle, math
from PIL import Image, ImageDraw, ImageFont
from geo import *
import ov
G=pickle.load(open('geo_data.pkl','rb'))
def en2z18(e,n):
    lon=np.asarray(e)/(111320*math.cos(math.radians(LAT0)))+LON0; lat=np.asarray(n)/110574+LAT0
    X,Y=ll2gpx(lon,lat,18); return X-69727*256, Y-97241*256
CROP=(450,520,1650,2620)
try: F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22); Fs=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
except: F=Fs=ImageFont.load_default()
def legend(im,title,items):
    d=ImageDraw.Draw(im,'RGBA'); h=40+26*len(items)
    d.rectangle([10,10,560,10+h],fill=(0,0,0,170)); d.text((20,16),title,font=F,fill='white')
    for i,(col,t) in enumerate(items):
        y=50+26*i; d.rectangle([22,y+4,48,y+16],fill=col); d.text((58,y),t,font=Fs,fill='white')
    # scale bar 100 m
    W,H=im.size; L=100/RES18; d.rectangle([W-L-30,H-50,W-30,H-40],fill='white'); d.text((W-L-30,H-36),'100 m',font=Fs,fill='white')
    # north arrow
    d.polygon([(W-50,30),(W-60,60),(W-40,60)],fill='white'); d.text((W-56,62),'N',font=Fs,fill='white')
def line(d,e,n,col,w,closed=True):
    x,y=en2z18(e,n); pts=list(zip(x-CROP[0],y-CROP[1]))
    if closed: pts.append(pts[0])
    d.line(pts,fill=col,width=w,joint='curve')
# (a) official map over satellite
ov.make(AX,AY,BEAR,1.0,out='/tmp/a_full.png')
a=Image.open('/tmp/a_full.png').crop(CROP).convert('RGB')
d=ImageDraw.Draw(a,'RGBA'); o=G['off']; line(d,o[:,0],o[:,1],(255,40,40,230),4)
legend(a,'FSAE Michigan 2026 Endurance map on Esri imagery (Jun 2019)',[((255,255,0),'Official map linework (roads, cones, fences)'),((255,40,40),'Official racing line, traced (~2.20 km lap)')])
a.save('/workspace/fsae_overlay/overlay_a_official_on_satellite.png')
# (b) AC over satellite
sat=Image.open('mis_z18.jpg').crop(CROP).convert('RGB'); d=ImageDraw.Draw(sat,'RGBA')
def poly(geom,fill,outline,w=3):
    gs=[geom] if geom.geom_type=='Polygon' else list(geom.geoms)
    for g in gs:
        x,y=en2z18(*np.array(g.exterior.coords).T); pts=list(zip(x-CROP[0],y-CROP[1]))
        d.polygon(pts,fill=fill); d.line(pts+[pts[0]],fill=outline,width=w)
poly(G['phys'],(0,200,255,40),(0,200,255,220),2)
poly(G['road'],(255,0,255,60),(255,0,255,255),3)
line(d,o[:,0],o[:,1],(255,40,40,240),4)
c=G['cones']; x,y=en2z18(c[:,0],c[:,1])
for xi,yi,m in zip(x,y,G['cone_mat']):
    col=(255,150,0) if 'Orange' in m else (40,90,255)
    d.ellipse([xi-CROP[0]-3,yi-CROP[1]-3,xi-CROP[0]+3,yi-CROP[1]+3],fill=col)
legend(sat,'Assetto Corsa fsae_michigan18.kn5 vs official 2026 line',[((255,0,255),'AC visual road mesh ROAD_4 (Tarmac)'),((0,200,255),'AC physics road mesh 1ROAD_3'),((255,150,0),'AC cones (494 orange + 2 blue)'),((255,40,40),'Official 2026 Endurance line')])
sat.save('/workspace/fsae_overlay/overlay_b_ac_vs_official_on_satellite.png')
print('ok')
# --- extras: walls + gates on (b), and an overview on z17
mats,meshes=pickle.load(open('meshes.pkl','rb')); spots=pickle.load(open('spots.pkl','rb'))
b=Image.open('/workspace/fsae_overlay/overlay_b_ac_vs_official_on_satellite.png'); d=ImageDraw.Draw(b,'RGBA')
wallpts=[]
for m in meshes:
    if mats[m['mat']]['name']=='Cement001':
        lon,lat=ac2ll(m['pos'][:,0],m['pos'][:,2]); wallpts.append((lon,lat))
        X,Y=ll2gpx(lon,lat,18); X-=69727*256; Y-=97241*256
        for xi,yi in zip(X,Y): d.point((xi-CROP[0],yi-CROP[1]),fill=(230,230,230))
for k,(x,z) in spots.items():
    if k.endswith('_L') or 'HOTLAP' in k:
        lon,lat=ac2ll(x,z); X,Y=ll2gpx(lon,lat,18); X-=69727*256+CROP[0]; Y-=97241*256+CROP[1]
        d.regular_polygon((X,Y,8),5,fill=(255,255,255)); d.text((X+10,Y-8),k.replace('AC_','').replace('_L',''),font=Fs,fill='white')
b.save('/workspace/fsae_overlay/overlay_b_ac_vs_official_on_satellite.png')
# overview on z17
o17=Image.open('mis_z17.jpg').convert('RGB'); d=ImageDraw.Draw(o17,'RGBA')
def to17(lon,lat):
    X,Y=ll2gpx(lon,lat,17); return X-34863*256, Y-48620*256
for lon,lat in wallpts:
    X,Y=to17(lon,lat)
    for xi,yi in zip(X,Y): d.ellipse([xi-1.5,yi-1.5,xi+1.5,yi+1.5],fill=(255,255,0))
for name,col in [('road',(255,0,255,255))]:
    g=G[name]; x,y=en2z18(*np.array(g.exterior.coords).T); lon,lat=gpx2ll(x+69727*256,y+97241*256,18); X,Y=to17(lon,lat)
    d.line(list(zip(X,Y))+[(X[0],Y[0])],fill=col,width=3)
lon,lat=G['off'][:,2],G['off'][:,3]; X,Y=to17(lon,lat); d.line(list(zip(X,Y))+[(X[0],Y[0])],fill=(255,40,40),width=3)
dd=ImageDraw.Draw(o17,'RGBA'); dd.rectangle([10,10,640,95],fill=(0,0,0,170))
dd.text((20,16),'Georeference check: AC walls (yellow) vs real MIS oval',font=F,fill='white')
dd.text((20,46),'magenta = AC road slab, red = official 2026 Endurance line',font=Fs,fill='white')
dd.text((20,68),'Esri World Imagery z17, GeoEye-1 30 Jun 2019',font=Fs,fill='white')
o17.save('/workspace/fsae_overlay/overlay_c_georef_check_walls.png')

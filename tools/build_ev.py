import numpy as np, pickle, json, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from geo import *
from evov import evcoef, ev2z18
CO=evcoef(0,113,1.0,13.5)
def ev2ll(u,v):
    x,y=ev2z18(u,v,CO); return gpx2ll(69727*256+x,97241*256+y,18)
G=pickle.load(open('geo_data.pkl','rb')); mats,meshes=pickle.load(open('meshes.pkl','rb')); spots=pickle.load(open('spots.pkl','rb'))
cl=np.load('ev_cl_px.npy'); lon,lat=ev2ll(cl[:,0],cl[:,1]); e,n=ll2en(lon,lat)
ev=LineString(np.c_[e,n]); road=G['road']; phys=G['phys']
st={}
st['ev_len']=ev.length; st['in_road']=ev.intersection(road).length; st['in_phys']=ev.intersection(phys).length
cones=G['cones'][:,:2]; d=np.array([ev.distance(Point(p)) for p in cones]); st['cones_10m']=float((d<10).mean()); st['cone_med']=float(np.median(d))
evpoly=Polygon(np.c_[e,n]).buffer(0); r=np.array(evpoly.minimum_rotated_rectangle.exterior.coords); st['ev_rect']=[float(np.hypot(*(r[i+1]-r[i]))) for i in range(2)]
# distance of EV line points outside road
outside=[ev.interpolate(t).distance(road) for t in np.arange(0,ev.length,1.0)]
st['max_out_m']=float(max(outside)); st['frac_pts_out']=float(np.mean(np.array(outside)>0))
# along-track coverage: project onto straight axis
ax=np.array([math.sin(math.radians(13.5)),math.cos(math.radians(13.5))])
pe=np.c_[e,n]@ax; pc=cones@ax; st['ev_span']=(float(pe.min()),float(pe.max())); st['cone_span']=(float(pc.min()),float(pc.max()))
rc=np.array(road.exterior.coords)@ax; st['road_span']=(float(rc.min()),float(rc.max()))
print(json.dumps(st,indent=1))
pickle.dump(dict(e=e,n=n,lon=lon,lat=lat,stats=st),open('ev_geo.pkl','wb'))
# ---------- overlays
def en2z18(e,n):
    lo=np.asarray(e)/(111320*math.cos(math.radians(LAT0)))+LON0; la=np.asarray(n)/110574+LAT0
    X,Y=ll2gpx(lo,la,18); return X-69727*256, Y-97241*256
gx,gy=ev2z18(np.array([0,1600,0,1600]),np.array([0,0,191,191]),CO)
CROP=(int(gx.min())-60,int(gy.min())-60,int(gx.max())+60,int(gy.max())+60); print('crop',CROP)
F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22); Fs=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
def legend(im,title,items):
    d=ImageDraw.Draw(im,'RGBA'); h=40+26*len(items)
    d.rectangle([10,10,10+max(560,13*len(title)),10+h],fill=(0,0,0,175)); d.text((20,16),title,font=F,fill='white')
    for i,(col,t) in enumerate(items):
        y=50+26*i; d.rectangle([22,y+4,48,y+16],fill=col); d.text((58,y),t,font=Fs,fill='white')
    W,H=im.size; L=100/RES18; d.rectangle([W-L-30,H-50,W-30,H-40],fill='white'); d.text((W-L-30,H-36),'100 m',font=Fs,fill='white')
    d.polygon([(W-50,30),(W-60,60),(W-40,60)],fill='white'); d.text((W-56,62),'N',font=Fs,fill='white')
sat=Image.open('mis_z18.jpg').convert('RGBA')
# warp EV map linework
mp=np.array(Image.open('ev_official.png').convert('RGBA')).astype(int); L=mp[...,:3].mean(2)
red=(mp[...,0]>180)&(mp[...,1]<110)&(mp[...,2]<110)
a=np.where(red|(L<90),255,0); rgb=mp[...,:3].copy(); rgb[~red]=[255,255,0]
mi=Image.fromarray(np.dstack([rgb,a]).astype(np.uint8))
A,B,C,D,E,Fc=CO; Mi=np.linalg.inv(np.array([[A,B,C],[D,E,Fc],[0,0,1]]))
w=mi.transform(sat.size,Image.AFFINE,tuple(Mi[:2].ravel()),resample=Image.BILINEAR)
al=w.getchannel('A').filter(ImageFilter.MaxFilter(3)); w.putalpha(al)
aimg=Image.alpha_composite(sat,w).crop(CROP).convert('RGB'); d=ImageDraw.Draw(aimg,'RGBA')
x,y=en2z18(e,n); pts=list(zip(x-CROP[0],y-CROP[1])); d.line(pts+[pts[0]],fill=(255,40,40,230),width=4)
legend(aimg,'FSAE EV 2026 Endurance on Esri imagery (2019)',[((255,255,0),'Official EV map linework (grid 30 ft, cones, gates)'),((255,40,40),f'EV racing line, traced (~{ev.length/1000:.2f} km lap)')])
aimg.save('/workspace/fsae_overlay/overlay_a_official_on_satellite.png')
# (b)
bimg=Image.open('mis_z18.jpg').crop(CROP).convert('RGB'); d=ImageDraw.Draw(bimg,'RGBA')
def poly(geom,fill,outline,wd=3):
    for g in ([geom] if geom.geom_type=='Polygon' else list(geom.geoms)):
        x,y=en2z18(*np.array(g.exterior.coords).T); p=list(zip(x-CROP[0],y-CROP[1])); d.polygon(p,fill=fill); d.line(p+[p[0]],fill=outline,width=wd)
poly(phys,(0,200,255,40),(0,200,255,220),2); poly(road,(255,0,255,55),(255,0,255,255),3)
d.line(pts+[pts[0]],fill=(255,40,40,240),width=4)
x,y=en2z18(cones[:,0],cones[:,1])
for xi,yi,m in zip(x,y,G['cone_mat']):
    d.ellipse([xi-CROP[0]-3,yi-CROP[1]-3,xi-CROP[0]+3,yi-CROP[1]+3],fill=(255,150,0) if 'Orange' in m else (40,90,255))
for k,(sx,sz) in spots.items():
    if k.endswith('_L') or 'HOTLAP' in k:
        lo,la=ac2ll(sx,sz); X,Y=ll2gpx(lo,la,18); X-=69727*256+CROP[0]; Y-=97241*256+CROP[1]
        d.regular_polygon((X,Y,8),5,fill='white'); d.text((X+10,Y-8),k.replace('AC_','').replace('_L',''),font=Fs,fill='white')
legend(bimg,'AC fsae_michigan18 vs FSAE EV 2026 Endurance',[((255,0,255),'AC visual road ROAD_4'),((0,200,255),'AC physics road 1ROAD_3'),((255,150,0),'AC cones (494 orange + 2 blue)'),((255,40,40),'Official EV 2026 Endurance line')])
bimg.save('/workspace/fsae_overlay/overlay_b_ac_vs_official_on_satellite.png')
# ---------- exports
R=lambda v:round(float(v),7)
line=LineString(np.c_[e,n]).simplify(0.3); lo,la=(np.array(line.coords)[:,0]/(111320*math.cos(math.radians(LAT0)))+LON0, np.array(line.coords)[:,1]/110574+LAT0)
coords=[[R(p),R(q)] for p,q in zip(lo,la)]; coords.append(coords[0])
feat=dict(type='Feature',properties=dict(name='Official FSAE Electric 2026 Endurance line (traced)',layer='official',length_m=round(ev.length),source='https://www.fsaeonline.com/app/MapViewer.aspx?mapid=5d2400e9-5197-4d84-a04d-e15895d83198'),geometry=dict(type='LineString',coordinates=coords))
json.dump(dict(type='FeatureCollection',features=[feat]),open('/workspace/fsae_overlay/official_line.geojson','w'))
ac=json.load(open('/workspace/fsae_overlay/ac_track.geojson'))
json.dump(dict(type='FeatureCollection',features=[feat]+[f for f in ac['features'] if f['properties']['layer'] in ('road','phys','cone','spot')]),open('/workspace/fsae_overlay/fsae_michigan_all.geojson','w'))
corners={k:[R(v) for v in ev2ll(u,v_)[::-1]] for k,(u,v_) in dict(tl=(0,0),tr=(1600,0),bl=(0,191)).items()}
src=open('/workspace/fsae_overlay/data.js').read()
icline=src.split('const OFFICIAL=')[1].split(';\n')[0]; iccorn=src.split('const MAPCORNERS=')[1].split(';\n')[0]
if 'Electric' in icline:  # rerun safety
    icline=src.split('const ICLINE=')[1].split(';\n')[0]; iccorn=src.split('const ICCORNERS=')[1].split(';\n')[0]
pivot=[R(v) for v in ac2ll(0,0)[::-1]]
js=('const OFFICIAL='+json.dumps(feat)+';\nconst ACTRACK='+json.dumps(ac)+';\nconst MAPCORNERS='+json.dumps(corners)+';\nconst ACPIVOT='+json.dumps(pivot)+
    ';\nconst ICLINE='+icline+';\nconst ICCORNERS='+iccorn+';\n')
open('/workspace/fsae_overlay/data.js','w').write(js)
def kc(cs): return ' '.join(f'{p},{q},0' for p,q in cs)
kml=['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>FSAE Electric 2026 Endurance vs AC fsae_michigan18</name>',
'<Style id="off"><LineStyle><color>ff2828ff</color><width>3</width></LineStyle></Style><Style id="ac"><LineStyle><color>ffff00ff</color><width>2</width></LineStyle><PolyStyle><color>40ff00ff</color></PolyStyle></Style><Style id="cone"><IconStyle><color>ff0096ff</color><scale>0.4</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
f'<Placemark><name>FSAE Electric 2026 Endurance line (traced from fsaeonline EV map, ~{round(ev.length)} m)</name><styleUrl>#off</styleUrl><LineString><tessellate>1</tessellate><coordinates>{kc(coords)}</coordinates></LineString></Placemark>']
for f in ac['features']:
    if f['properties']['layer']=='road': kml.append(f'<Placemark><name>AC ROAD_4 tarmac</name><styleUrl>#ac</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{kc(f["geometry"]["coordinates"][0])}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
kml.append('<Folder><name>AC cones</name><visibility>0</visibility>')
for f in ac['features']:
    if f['properties']['layer']=='cone': c=f['geometry']['coordinates']; kml.append(f'<Placemark><visibility>0</visibility><styleUrl>#cone</styleUrl><Point><coordinates>{c[0]},{c[1]},0</coordinates></Point></Placemark>')
kml.append('</Folder></Document></kml>')
open('/workspace/fsae_overlay/fsae_michigan_2026_endurance.kml','w').write('\n'.join(kml))
print(corners)

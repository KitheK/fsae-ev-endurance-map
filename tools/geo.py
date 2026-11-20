import numpy as np, math, pickle
MPP_MAP=0.3048*100/61.9
def gpx2ll(X,Y,z):
    n=256*2**z; lon=X/n*360-180
    lat=np.degrees(np.arctan(np.sinh(np.pi*(1-2*Y/n)))); return lon,lat
def ll2gpx(lon,lat,z):
    n=256*2**z; X=(lon+180)/360*n
    Y=(1-np.log(np.tan(np.radians(lat))+1/np.cos(np.radians(lat)))/np.pi)/2*n; return X,Y
# official map (1600px) -> z18 mosaic px
AX,AY,BEAR=1371,919,13.5
RES18=0.4437
def map2z18(u,v,ax=AX,ay=AY,bear=BEAR,scale=1.0):
    b=math.radians(bear); k=MPP_MAP*scale/RES18
    rx,ry=math.sin(b),-math.cos(b); ux,uy=-math.cos(b),-math.sin(b)
    du=np.asarray(u)-1383; dv=np.asarray(v)-522.5
    return ax+k*(du*rx-dv*ux), ay+k*(du*ry-dv*uy)
def map2ll(u,v,**kw):
    x,y=map2z18(u,v,**kw); return gpx2ll(69727*256+x,97241*256+y,18)
# AC world -> lon/lat via embedded GoogleMap texture + SIFT fit to Esri z17
K=8192/7249.6
_M_GPath = __file__.replace("geo.py", "M_gm2z17.npy")
def ac2ll(x,z):
    import os
    path = os.environ.get("FSAE_M_GM2Z17", _M_GPath)
    if not os.path.isfile(path):
        path = "/workspace/fsae_overlay/src/M_gm2z17.npy"
    if not os.path.isfile(path):
        raise FileNotFoundError("M_gm2z17.npy required for ac2ll (set FSAE_M_GM2Z17)")
    M = np.load(path)
    tu=4096-np.asarray(x)*K; tv=4096-np.asarray(z)*K
    mx=M[0,0]*tu+M[0,1]*tv+M[0,2]; my=M[1,0]*tu+M[1,1]*tv+M[1,2]
    return gpx2ll(34863*256+mx,48620*256+my,17)
LAT0,LON0=42.0665,-84.2390
def ll2en(lon,lat):
    return (np.asarray(lon)-LON0)*111320*math.cos(math.radians(LAT0)), (np.asarray(lat)-LAT0)*110574

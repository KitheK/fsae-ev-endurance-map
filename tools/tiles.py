import math, os, sys, requests, io
from PIL import Image
def deg2num(lat, lon, z):
    n=2**z; x=(lon+180)/360*n; y=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n
    return x,y
def mosaic(lat0,lon0,lat1,lon1,z,out):
    x0,y0=deg2num(lat1,lon0,z); x1,y1=deg2num(lat0,lon1,z)
    X0,Y0,X1,Y1=int(x0),int(y0),int(x1),int(y1)
    W=(X1-X0+1)*256;H=(Y1-Y0+1)*256
    im=Image.new('RGB',(W,H))
    s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0'
    for tx in range(X0,X1+1):
        for ty in range(Y0,Y1+1):
            fn=f'tiles/{z}_{tx}_{ty}.jpg'
            if not os.path.exists(fn):
                os.makedirs('tiles',exist_ok=True)
                r=s.get(f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty}/{tx}',timeout=30)
                open(fn,'wb').write(r.content)
            try: t=Image.open(fn).convert('RGB')
            except: continue
            im.paste(t,((tx-X0)*256,(ty-Y0)*256))
    im.save(out); print(out,im.size,'origin tile',X0,Y0)
    return X0,Y0
if __name__=='__main__':
    a=list(map(float,sys.argv[1:5])); mosaic(*a,int(sys.argv[5]),sys.argv[6])

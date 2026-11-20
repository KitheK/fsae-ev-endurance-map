import struct, sys, numpy as np
class R:
    def __init__(s,b): s.b=b; s.p=0
    def i(s): v=struct.unpack_from('<i',s.b,s.p)[0]; s.p+=4; return v
    def u8(s): v=s.b[s.p]; s.p+=1; return v
    def f(s,n=1):
        v=struct.unpack_from('<%df'%n,s.b,s.p); s.p+=4*n; return v if n>1 else v[0]
    def s(s_):
        n=s_.i(); v=s_.b[s_.p:s_.p+n].decode('latin1'); s_.p+=n; return v
def parse(path):
    b=open(path,'rb').read(); r=R(b)
    assert b[:6]==b'sc6969'; r.p=6; ver=r.i()
    if ver>5: r.i()
    tex=[]
    for _ in range(r.i()):
        act=r.i(); name=r.s(); sz=r.i(); tex.append((name,sz)); r.p+=sz
    mats=[]
    for _ in range(r.i()):
        name=r.s(); shader=r.s(); r.u8()
        if ver>4: r.u8()
        r.i()
        props={}
        for _ in range(r.i()):
            pn=r.s(); pv=r.f(); r.p+=36; props[pn]=pv
        samp=[]
        for _ in range(r.i()):
            sn=r.s(); r.i(); st=r.s(); samp.append((sn,st))
        mats.append(dict(name=name,shader=shader,samplers=samp))
    meshes=[]
    def node(M,path):
        t=r.i(); name=r.s(); nc=r.i(); r.u8()
        if t==1:
            m=np.array(r.f(16)).reshape(4,4)  # row-major, translation in row 3
            M=m@M
        elif t==2:
            r.u8();r.u8();r.u8()
            vc=r.i(); v=np.frombuffer(b,dtype='<f4',count=vc*11,offset=r.p).reshape(vc,11); r.p+=vc*44
            ic=r.i(); idx=np.frombuffer(b,dtype='<u2',count=ic,offset=r.p); r.p+=ic*2
            mid=r.i(); r.i(); r.f(2); r.f(4); r.u8()
            pos=np.c_[v[:,:3],np.ones(vc)]@M
            meshes.append(dict(name=name,path=path,pos=pos[:,:3],uv=v[:,6:8].copy(),idx=idx.copy(),mat=mid))
        elif t==3:
            r.u8();r.u8();r.u8()
            for _ in range(r.i()): r.s(); r.f(16)
            vc=r.i(); r.p+=vc*76; ic=r.i(); r.p+=ic*2; r.i(); r.i(); r.f(2)
        else: raise ValueError('node type %d at %d'%(t,r.p))
        for _ in range(nc): node(M,path+'/'+name)
    node(np.eye(4),'')
    return ver,tex,mats,meshes
if __name__=='__main__':
    ver,tex,mats,meshes=parse(sys.argv[1])
    print('version',ver,'textures',len(tex),'materials',len(mats),'meshes',len(meshes))
    for n,sz in sorted(tex,key=lambda x:-x[1])[:8]: print(' tex',n,sz)
    for m in meshes:
        p=m['pos']; e=p.max(0)-p.min(0)
        print(f"{m['name']:40s} v={len(p):7d} mat={mats[m['mat']]['name'][:25]:25s} ext x={e[0]:7.1f} y={e[1]:6.1f} z={e[2]:7.1f}")

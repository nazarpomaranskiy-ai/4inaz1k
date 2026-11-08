"""Narrow detector for red/yellow fire bolts, with registered motion evidence.

This intentionally does not claim other attacks. It has no patch RF fallback.
"""
import cv2
import json
from pathlib import Path
from imaging import read_image
from planner import Shot
import math
import numpy as np

def appearance_descriptor(image):
    hsv=cv2.cvtColor(image[:,:,:3],cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    red=(((h<=14)|(h>=173))&(s>=130)&(v>=145)).astype(np.uint8)
    yellow=((h>=15)&(h<=39)&(s>=130)&(v>=160)).astype(np.uint8)
    if image.shape[2]==4:
        alpha=image[:,:,3]>64;red*=alpha;yellow*=alpha
    yy,xx=np.where(red|yellow)
    if len(xx)<4:return None
    box=(slice(yy.min(),yy.max()+1),slice(xx.min(),xx.max()+1))
    vector=np.stack([cv2.resize(c[box].astype(np.float32),(24,24)) for c in (red,yellow)]).ravel()
    return vector/max(1e-6,float(np.linalg.norm(vector)))

class RealmPilot:
    def __init__(self,cfg,root=None):
        self.cfg=cfg;self.previous=None;self.unknown=[];self.overflow=False
        self.status='PILOT: Fire Missile only';self.candidates=[];self.scene_supported=False
        self.bank=[];self.marker=None;self.raw_previous=[];self.clock=0;self.motion_shots=[]
        if root is not None:
            root=Path(root);self.marker=read_image(root/'pilot_assets/realm_score.png',cv2.IMREAD_GRAYSCALE)
            path=root/'game_data/catalog.json'
            if path.exists():
                catalog=json.loads(path.read_text('utf-8'))
                entry=next((o for o in catalog['objects'] if o['id']=='Fire Missile' and o['sprites']),None)
                if entry:
                    image=read_image(root/'game_data/sprites'/entry['sprites'][0],cv2.IMREAD_UNCHANGED)
                    hh,ww=image.shape[:2];size=max(hh,ww)*2
                    canvas=np.zeros((size,size,4),np.uint8)
                    canvas[(size-hh)//2:(size-hh)//2+hh,(size-ww)//2:(size-ww)//2+ww]=image
                    for angle in range(0,360,45):
                        rotated=cv2.warpAffine(canvas,cv2.getRotationMatrix2D((size/2,size/2),angle,1),(size,size),flags=cv2.INTER_NEAREST)
                        descriptor=appearance_descriptor(rotated)
                        if descriptor is not None:self.bank.append(descriptor)
        self.bank=np.array(self.bank,np.float32)
        self.require_reference=bool(cfg.get('pilot_require_reference',True))
    def detect(self,frame,shift=(0,0),camera_confidence=0,hints=(),dt=1/15):
        self.clock+=dt;self.motion_shots=[]
        raw=[]
        self.scene_supported=False
        if self.marker is not None:
            roi=cv2.cvtColor(frame[:140,:380],cv2.COLOR_BGR2GRAY)
            if roi.shape[0]>=self.marker.shape[0] and roi.shape[1]>=self.marker.shape[1]:
                self.scene_supported=float(cv2.matchTemplate(roi,self.marker,cv2.TM_CCOEFF_NORMED).max())>=.85
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        red=(((h<=12)|(h>=174))&(s>=145)&(v>=155)).astype(np.uint8)
        yellow=((h>=14)&(h<=38)&(s>=145)&(v>=185)).astype(np.uint8)
        colors=((red|yellow)*255).astype(np.uint8)
        mask=cv2.morphologyEx(colors,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        moved=np.zeros(frame.shape[:2],np.uint8)
        if self.previous is not None and camera_confidence>=.6:
            dx,dy=shift
            previous=cv2.warpAffine(self.previous,np.float32([[1,0,dx],[0,1,dy]]),
                                    (frame.shape[1],frame.shape[0]),flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_REFLECT)
            moved=(cv2.absdiff(frame,previous).max(axis=2)>50).astype(np.uint8)
            border=int(max(abs(dx),abs(dy)))+3
            if border:
                moved[:border]=0;moved[-border:]=0;moved[:,:border]=0;moved[:,-border:]=0
        self.previous=frame.copy()
        found=[];appearance=[]
        l,t,r,b=self.cfg.get('arena',[0,0,frame.shape[1],frame.shape[0]])
        for i in range(1,count):
            x,y,w,hh,area=stats[i]
            if not (5<=area<=600 and 4<=max(w,hh)<=40 and min(w,hh)>=2):continue
            component=labels[y:y+hh,x:x+w]==i
            reds=int(np.sum(red[y:y+hh,x:x+w][component]));yellows=int(np.sum(yellow[y:y+hh,x:x+w][component]))
            if reds<4 or yellows<2 or reds<yellows*.15 or yellows<reds*.08:continue
            score=1.0;angle=0
            if len(self.bank):
                matches=[]
                for pad in (0,2,8,12):
                    descriptor=appearance_descriptor(frame[max(0,y-pad):y+hh+pad,max(0,x-pad):x+w+pad])
                    if descriptor is not None:
                        values=(self.bank*descriptor).sum(axis=1);matches.append((float(values.max()),int(values.argmax())*45))
                if not matches:continue
                score,angle=max(matches)
            elif self.require_reference:
                continue
            cx,cy=centers[i];radius=max(w,hh)/2
            if not l<=cx<r or not t<=cy<b:continue
            if any(x1<=cx<x2 and y1<=cy<y2 for x1,y1,x2,y2 in self.cfg.get('ignore_rectangles',[])):continue
            item=dict(x=float(cx),y=float(cy),r=float(radius),score=score,angle=angle,
                      trusted_until=self.clock+.3 if score>=.8 else 0)
            raw.append(item)
            # Match a verified sprite to the previous frame's COLOR candidates,
            # even if that previous appearance was too weak for classification.
            pairs=[]
            if camera_confidence>=.6:
                for old in self.raw_previous:
                    dx=cx-old['x']-shift[0];dy=cy-old['y']-shift[1];distance=math.hypot(dx,dy)
                    speed=distance/max(.001,dt)
                    trusted=old['trusted_until']>=self.clock
                    if not (50<=speed<=1000 and old['score']>=.4 and (score>=.8 or trusted and score>=.45)):continue
                    orientation=angle if score>=.8 else old['angle']
                    heading=-math.pi/4-math.radians(orientation)
                    alignment=(dx*math.cos(heading)+dy*math.sin(heading))/max(.001,distance)
                    if alignment<.65:continue
                    cost=abs(distance-320*dt)+(1-alignment)*30
                    pairs.append((cost,old,dx/dt,dy/dt))
            if pairs:
                _,old,vx,vy=min(pairs,key=lambda value:value[0])
                item['trusted_until']=max(item['trusted_until'],old['trusted_until'])
                if score<.8:item['angle']=old['angle']
                self.motion_shots.append(Shot(float(cx),float(cy),float(vx),float(vy),float(radius),uncertainty=3))
            if score<.8 and not pairs:continue
            appearance.append((float(cx),float(cy),float(radius)))
            motion_fraction=float(np.mean(moved[y:y+hh,x:x+w][component]))
            if motion_fraction>=.35:found.append((float(cx),float(cy),float(radius)))
        self.raw_previous=raw
        self.candidates=appearance
        self.unknown=[c for c in appearance if c not in found]
        self.overflow=len(found)>80
        self.status=f'PILOT Fire Missile only: {len(found)} moving / {len(appearance)} matched; other attacks unsupported'
        if self.require_reference and not len(self.bank):self.status='PILOT reference missing: run prepare_learning.bat'
        return found,mask
    def close(self):pass

def confirmed_shots(tracker,shots):
    """Require sustained world motion, not animated scenery or one-frame color hits."""
    stable=[]
    for track in tracker.tracks:
        if len(track.history)<2 or track.missed>1:continue
        first,last=track.history[0],track.history[-1]
        distance=float(np.hypot(last[1]-first[1],last[2]-first[2]))
        if distance<6:continue
        stable.append(track.predict(tracker.stamp))
    return [shot for shot in shots if 50<=np.hypot(shot.vx,shot.vy)<=1200
            and any(np.linalg.norm(p-[shot.x,shot.y])<2 for p in stable)]


def merge_pilot_motion(tracker,shots,engine,offset):
    confirmed=confirmed_shots(tracker,shots)
    for direct in engine.motion_shots:
        position=np.array([direct.x+offset[0],direct.y+offset[1]])
        match=next((s for s in confirmed if np.hypot(s.x-position[0],s.y-position[1])<max(10,direct.radius)),None)
        if match is not None:
            match.vx=.5*match.vx+.5*direct.vx;match.vy=.5*match.vy+.5*direct.vy
        else:
            confirmed.append(Shot(float(position[0]),float(position[1]),direct.vx,direct.vy,direct.radius,uncertainty=direct.uncertainty))
    return confirmed

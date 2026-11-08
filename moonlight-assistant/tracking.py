"""Online trajectory fitting with pre-update prediction error and camera flow."""
from collections import deque
from dataclasses import dataclass,field
import math
import cv2
import numpy as np
from planner import Shot

class CameraMotion:
    def __init__(self,player):
        self.player=np.array(player,dtype=float);self.previous=None;self.offset=np.zeros(2)
        self.confidence=0.0;self.rotated=False;self.shift=np.zeros(2)
    def update(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        image_scale=min(1,640/max(gray.shape))
        if image_scale<1:gray=cv2.resize(gray,None,fx=image_scale,fy=image_scale,interpolation=cv2.INTER_AREA)
        if self.previous is None or self.previous.shape!=gray.shape:
            self.previous=gray;self.confidence=0;return self.offset.copy()
        points=cv2.goodFeaturesToTrack(self.previous,maxCorners=160,qualityLevel=.03,minDistance=12)
        self.confidence=0;self.rotated=False;self.shift=np.zeros(2)
        if points is not None and len(points)>=12:
            moved,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None,
                                                 winSize=(21,21),maxLevel=3)
            if moved is not None:
                source=points[status.ravel()==1].reshape(-1,2)
                dest=moved[status.ravel()==1].reshape(-1,2)
                if len(source)>=12:
                    matrix,inliers=cv2.estimateAffinePartial2D(source,dest,method=cv2.RANSAC,
                                                              ransacReprojThreshold=1.8)
                    if matrix is not None:
                        ratio=float(np.mean(inliers)); angle=math.atan2(matrix[1,0],matrix[0,0])
                        transform_scale=math.hypot(matrix[0,0],matrix[1,0])
                        self.rotated=abs(angle)>.015 or abs(transform_scale-1)>.025
                        if ratio>.6 and not self.rotated:
                            shift=(matrix[:,:2]@(self.player*image_scale)+matrix[:,2]-self.player*image_scale)/image_scale
                            if np.linalg.norm(shift)<100:
                                self.offset-=shift;self.shift=shift;self.confidence=ratio
        self.previous=gray
        return self.offset.copy()

@dataclass
class Track:
    identity:int
    radius:float
    history:deque=field(default_factory=lambda:deque(maxlen=12))
    coefficients:dict=field(default_factory=dict)
    errors:dict=field(default_factory=lambda:{1:4.0,2:4.0})
    model:int=1
    missed:int=0
    def predict(self,stamp,order=None):
        order=order or self.model
        if order not in self.coefficients:return np.array(self.history[-1][1:])
        coeff=self.coefficients[order];t=stamp-self.history[-1][0]
        return np.array(self.history[-1][1:])+coeff[1]*t+(coeff[2]*t*t if len(coeff)>2 else 0)
    def update(self,stamp,x,y,radius):
        # Score the OLD models before observing the new position: no leakage.
        if self.history:
            for order in self.coefficients:
                err=float(np.linalg.norm(self.predict(stamp,order)-[x,y]))
                self.errors[order]=.8*self.errors[order]+.2*min(80,err)
        self.history.append((stamp,x,y));self.radius=radius;self.missed=0
        values=np.array(self.history);t=values[:,0]-stamp
        if len(values)>=2:
            for order in (1,2):
                if len(values)<(5 if order==2 else 2):continue
                design=np.stack([t**power for power in range(order+1)],axis=1)
                coeff=np.linalg.lstsq(design,values[:,1:],rcond=None)[0]
                if order==2 and np.linalg.norm(coeff[2])*2>5000:continue
                self.coefficients[order]=coeff
        self.model=2 if len(values)>=7 and 2 in self.coefficients and self.errors[2]<self.errors[1]*.85 else 1

class AdaptiveTracker:
    def __init__(self,gate=35,max_speed=1400,retain_seconds=.10):
        self.gate=gate;self.max_speed=max_speed;self.tracks=[];self.counter=0;self.stamp=0
        self.retain_seconds=retain_seconds
    def update(self,detections,dt):
        self.stamp+=dt;stamp=self.stamp
        pairs=[]
        if self.tracks and detections:
            positions=np.array([t.predict(stamp) for t in self.tracks])
            obs=np.array(detections)
            distances=np.linalg.norm(positions[:,None,:]-obs[None,:,:2],axis=2)
            radii=np.array([t.radius for t in self.tracks])
            gates=np.array([self.gate+t.missed*8 for t in self.tracks])
            allowed=(distances<gates[:,None])&(np.abs(radii[:,None]-obs[None,:,2])<np.maximum(6,radii[:,None]))
            ii,jj=np.where(allowed)
            pairs=[(float(distances[i,j]),int(i),int(j)) for i,j in zip(ii,jj)]
        used_i=set();used_j=set()
        for distance,i,j in sorted(pairs):
            if i in used_i or j in used_j:continue
            track=self.tracks[i];x,y,r=detections[j]
            previous=track.history[-1]
            if math.hypot(x-previous[1],y-previous[2])/max(.001,stamp-previous[0])>self.max_speed:continue
            track.update(stamp,x,y,r);used_i.add(i);used_j.add(j)
        for i,track in enumerate(self.tracks):
            if i not in used_i:track.missed+=1
        self.tracks=[t for t in self.tracks if stamp-t.history[-1][0]<=self.retain_seconds and t.missed<=3]
        for j,(x,y,r) in enumerate(detections):
            if j not in used_j:
                track=Track(self.counter,r);self.counter+=1;track.update(stamp,x,y,r);self.tracks.append(track)
        shots=[]
        for track in self.tracks:
            if len(track.history)<2:continue
            coeff=track.coefficients.get(track.model)
            if coeff is None:continue
            position=track.predict(stamp)
            age=stamp-track.history[-1][0]
            acceleration=2*coeff[2] if track.model==2 else np.zeros(2)
            velocity=coeff[1]+acceleration*age
            uncertainty=track.errors[track.model]+track.missed*4
            shots.append(Shot(*position,*velocity,track.radius,*acceleration,uncertainty))
        return shots

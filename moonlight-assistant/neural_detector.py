"""One-class YOLO Fire Missile detector. Other projectiles are unsupported."""
from pathlib import Path
import cv2
import numpy as np
from imaging import read_image

class NeuralFireDetector:
    def __init__(self,cfg,root):
        root=Path(root);self.cfg=cfg;self.status='Loading Fire Missile network'
        self.unknown=[];self.motion_shots=[];self.overflow=False;self.scene_supported=False
        self.marker=read_image(root/'pilot_assets/realm_score.png',cv2.IMREAD_GRAYSCALE)
        weights=root/'models/neural_fire.pt'
        if not weights.exists():raise FileNotFoundError('Validated Fire Missile weights are missing')
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(2)
        self.device=0 if torch.cuda.is_available() else 'cpu'
        self.model=YOLO(str(weights))
        names=list(self.model.names.values()) if isinstance(self.model.names,dict) else self.model.names
        if names!=['Fire Missile']:raise ValueError('Wrong network: expected one Fire Missile class')
        self.bank=[True]  # Shared readiness interface with the retired template pilot.
        self.threshold=float(cfg.get('neural_confidence',.5))
        self.image_size=int(cfg.get('neural_image_size',1600))
        self.model.predict(np.zeros((720,1555,3),np.uint8),device=self.device,imgsz=self.image_size,
                           conf=self.threshold,verbose=False)
        self.boxes=[]
    def detect(self,frame,**unused):
        self.scene_supported=False
        if self.marker is not None:
            gray=cv2.cvtColor(frame[:140,:380],cv2.COLOR_BGR2GRAY)
            if gray.shape[0]>=self.marker.shape[0] and gray.shape[1]>=self.marker.shape[1]:
                self.scene_supported=float(cv2.matchTemplate(gray,self.marker,cv2.TM_CCOEFF_NORMED).max())>=.85
        result=self.model.predict(frame,device=self.device,imgsz=self.image_size,
                                  conf=self.threshold,iou=.45,max_det=100,verbose=False)[0]
        values=result.boxes.data.detach().cpu().numpy() if result.boxes is not None else np.empty((0,6))
        found=[];self.boxes=[]
        for x1,y1,x2,y2,score,kind in values:
            width,height=x2-x1,y2-y1
            if int(kind)!=0 or min(width,height)<2 or max(width,height)>80:continue
            cx,cy=float((x1+x2)/2),float((y1+y2)/2)
            l,t,r,b=self.cfg.get('arena',[0,0,frame.shape[1],frame.shape[0]])
            if not l<=cx<r or not t<=cy<b:continue
            if any(a<=cx<c and bb<=cy<d for a,bb,c,d in self.cfg.get('ignore_rectangles',[])):continue
            found.append((cx,cy,float(max(width,height)/2)))
            self.boxes.append([float(x1),float(y1),float(x2),float(y2),float(score)])
        self.overflow=len(found)>80
        self.status=f'NEURAL Fire Missile ONLY: {len(found)} detections; other attacks unsupported'
        return found,np.zeros(frame.shape[:2],np.uint8)
    def close(self):
        self.model=None

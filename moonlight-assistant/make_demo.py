"""Generate a synthetic clip to exercise the pipeline, not game validation."""
from pathlib import Path
import cv2
import numpy as np
out = cv2.VideoWriter(str(Path(__file__).with_name('demo.mp4')),
                      cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
if not out.isOpened():
    raise RuntimeError('Cannot open video encoder')
for i in range(150):
    frame = np.zeros((720,1280,3), np.uint8)
    cv2.circle(frame,(640,360),6,(255,255,255),-1)
    for j in range(7):
        cv2.circle(frame,(int(350+i*2.5),270+j*30),4,(0,0,255),-1)
    out.write(frame)
out.release()

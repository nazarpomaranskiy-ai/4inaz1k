"""Offline, configurable prototype. Does not send keyboard or mouse input."""
import argparse
import csv
import json
from pathlib import Path
import math
import cv2
import numpy as np
from planner import Shot, recommend
from learning import VisionEngine
from tracking import AdaptiveTracker,CameraMotion

class Tracker:
    def __init__(self, gate, max_speed):
        self.tracks = []
        self.gate = gate
        self.max_speed = max_speed

    def update(self, detections, dt):
        # One-to-one nearest predicted position matching. Discard unmatched tracks:
        # an occluded shot cannot safely be extrapolated indefinitely.
        pairs = []
        for i, old in enumerate(self.tracks):
            x, y, radius, vx, vy, age = old
            for j, (nx, ny, nr) in enumerate(detections):
                distance = math.hypot(nx - x - vx*dt, ny - y - vy*dt)
                if distance < self.gate and abs(nr-radius) < max(4, radius):
                    pairs.append((distance, i, j))
        used_i, used_j, updated = set(), set(), []
        for _, i, j in sorted(pairs):
            if i in used_i or j in used_j:
                continue
            x, y, _, vx, vy, age = self.tracks[i]
            nx, ny, nr = detections[j]
            ux, uy = (nx-x)/dt, (ny-y)/dt
            if math.hypot(ux, uy) > self.max_speed:
                continue
            used_i.add(i)
            used_j.add(j)
            # Initialize velocity without smoothing; smooth subsequent estimates.
            a = 1 if age == 1 else .6
            updated.append((nx, ny, nr, a*ux+(1-a)*vx, a*uy+(1-a)*vy, age+1))
        updated.extend((x, y, r, 0, 0, 1) for j, (x, y, r) in
                       enumerate(detections) if j not in used_j)
        self.tracks = updated
        return [Shot(x, y, vx, vy, r) for x, y, r, vx, vy, age in updated if age >= 3]

def detect(frame, cfg):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in cfg['bullet_hsv']:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    l, t, r, b = cfg['arena']
    allowed = np.zeros_like(mask)
    allowed[t:b, l:r] = 255
    mask &= allowed
    for x1, y1, x2, y2 in cfg.get('ignore_rectangles', []):
        mask[y1:y2, x1:x2] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if cfg['min_area'] <= area <= cfg['max_area']:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detections.append((x, y, radius))
    return detections, mask

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--learned',action='store_true',help='Use locally trained detector and adaptive trajectories')
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--output', type=Path, default=Path('annotated.mp4'))
    p.add_argument('--snapshot', type=Path, help='Extract first frame and exit for calibration')
    args = p.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        p.error('Cannot open input video')
    ok, first = cap.read()
    if not ok:
        p.error('Video contains no readable frames')
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), first):
            p.error('Cannot write snapshot')
        cap.release()
        return
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    h, w = first.shape[:2]
    if cfg['frame_size'] != [w, h]:
        p.error(f'Calibration is for {cfg["frame_size"]}; video is {[w,h]}. Edit config.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        p.error('Video has invalid FPS')
    l, t, r, b = cfg['arena']
    px, py = cfg['player']
    if not (0 <= l < r <= w and 0 <= t < b <= h and l <= px < r and t <= py < b):
        p.error('Invalid arena/player coordinates')
    if args.output.resolve() == args.video.resolve():
        p.error('Output must differ from input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    if not out.isOpened():
        p.error('Cannot create MP4 output')
    vision=VisionEngine(dict(cfg,collect_learning=False),Path(__file__).parent) if args.learned else None
    if vision and vision.model.model is None:p.error('Train detector with prepare_learning.bat first')
    camera=CameraMotion(cfg['player'])
    tracker = (AdaptiveTracker if vision else Tracker)(cfg['tracking_gate'], cfg['max_shot_speed'])
    frame, index = first, 0
    try:
        with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
            log = csv.writer(f)
            log.writerow(['seconds', 'detections', 'tracked', 'direction', 'clearance_px'])
            while True:
                offset=camera.update(frame) if vision else (0,0)
                detections, mask = vision.detect(frame) if vision else detect(frame, cfg)
                tracked=tracker.update([(x+offset[0],y+offset[1],r) for x,y,r in detections],1/fps)
                shots=[Shot(s.x-offset[0],s.y-offset[1],s.vx,s.vy,s.radius,s.ax,s.ay,s.uncertainty) for s in tracked]
                best, _ = recommend(cfg['player'], shots, cfg['arena'], cfg['player_speed'],
                                    cfg['player_radius'], cfg['horizon'], cfg['margin'])
                # Always show raw detections: essential to spot false positives.
                for x,y,radius in detections:
                    cv2.circle(frame, (round(x),round(y)), round(radius), (0,220,255), 1)
                for s in shots:
                    cv2.line(frame, (round(s.x),round(s.y)),
                             (round(s.x+s.vx*.15),round(s.y+s.vy*.15)), (255,100,0), 1)
                cv2.rectangle(frame,(l,t),(r,b),(180,180,180),1)
                cv2.circle(frame,(px,py),cfg['player_radius'],(255,255,255),1)
                text = 'UNKNOWN: insufficient tracks'
                direction, gap = '', ''
                if best:
                    direction, gap, vx, vy = best
                    text = f'{direction} | clearance {gap:.1f}px'
                    if gap <= 0:
                        text = 'NO SAFE CANDIDATE | ' + text
                    cv2.arrowedLine(frame,(px,py),
                                    (round(px+vx*.25), round(py+vy*.25)),
                                    (0,255,0) if gap>0 else (0,0,255),2)
                cv2.rectangle(frame,(0,0),(w,65),(0,0,0),-1)
                cv2.putText(frame,text,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                cv2.putText(frame,'EXPERIMENTAL / fixed player & camera / offline forecast',
                            (12,52),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
                out.write(frame)
                log.writerow([round(index/fps,4),len(detections),len(shots),direction,gap])
                index += 1
                ok, frame = cap.read()
                if not ok:
                    break
    finally:
        if vision:vision.close()
        cap.release()
        out.release()
    print(f'Analyzed {index} frames: {args.output} and {args.output.with_suffix(".csv")}')

if __name__ == '__main__':
    main()

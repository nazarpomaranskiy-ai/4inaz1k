"""Train a local RTrees patch detector; report held-out synthetic performance.

Sprite identity is split before augmentation. Real labeled patches and source
frames are also split before augmentation. Synthetic metrics are not game recall.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import os
import random
import time
import cv2
import numpy as np
from learning import features
from imaging import read_image

HERE=Path(__file__).resolve().parent

def sprite_patch(path,rng,backgrounds):
    sprite=read_image(path,cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim!=3:raise ValueError(f'Invalid sprite {path}')
    if sprite.shape[2]==3:sprite=np.dstack([sprite,np.full(sprite.shape[:2],255,np.uint8)])
    h,w=sprite.shape[:2]
    # Keep complete sprite within a normalized patch, random orientation and scale.
    size=rng.randint(12,23)
    scale=size/max(w,h)
    sprite=cv2.resize(sprite,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
    layer=np.zeros((40,40,4),np.uint8)
    h,w=sprite.shape[:2];x=(40-w)//2;y=(40-h)//2
    layer[y:y+h,x:x+w]=sprite
    angle=rng.uniform(0,360)
    matrix=cv2.getRotationMatrix2D((20,20),angle,1)
    layer=cv2.warpAffine(layer,matrix,(40,40),flags=cv2.INTER_NEAREST)
    if backgrounds and rng.random()<.5:
        bg=backgrounds[rng.randrange(len(backgrounds))]
        h,w=bg.shape[:2]
        x=rng.randrange(max(1,w-40));y=rng.randrange(max(1,h-40))
        bg=cv2.resize(bg[y:y+40,x:x+40],(40,40))
    else:
        color=np.array([rng.randrange(35,210) for _ in range(3)])
        bg=np.clip(color+np.random.default_rng(rng.randrange(2**32)).normal(0,15,(40,40,3)),0,255).astype(np.uint8)
    alpha=layer[:,:,3:4].astype(np.float32)/255
    patch=(layer[:,:,:3]*alpha+bg*(1-alpha)).astype(np.uint8)
    return patch[5:35,5:35]

def split_sources(files,seed):
    files=sorted(files);random.Random(seed).shuffle(files)
    n=max(1,int(len(files)*.2))
    return files[n:],files[:n]

def build(positives,negatives,labels,backgrounds,count,seed):
    rng=random.Random(seed);x=[];y=[]
    if not positives or not negatives:raise ValueError('Need extracted projectile and non-projectile sprites')
    for i in range(count):
        if i and i%4000==0:print(f'Prepared {i}/{count} patches',flush=True)
        label=i%2
        pool=positives if label else negatives
        patch=sprite_patch(pool[(i//2)%len(pool)],rng,backgrounds)
        # Background-only negatives improve discrimination from scenery.
        if not label and i%4==0:
            if backgrounds:
                bg=backgrounds[rng.randrange(len(backgrounds))];h,w=bg.shape[:2]
                xx=rng.randrange(max(1,w-32));yy=rng.randrange(max(1,h-32))
                patch=bg[yy:yy+32,xx:xx+32]
            else:patch=np.full((24,24,3),rng.randrange(30,220),np.uint8)
        x.append(features(patch));y.append(label)
    # Human labels are real observations; repeat at modest weight, never pseudo-label.
    for label,path in labels:
        patch=read_image(path)
        if patch is None:continue
        for _ in range(max(4,min(32,2000//max(1,len(labels))))):x.append(features(patch));y.append(label)
    return np.stack(x),np.array(y,np.int32)

def metrics(model,x,y,threshold):
    votes=model.getVotes(x,0);col=np.flatnonzero(votes[0]==1)[0]
    prob=votes[1:,col]/np.maximum(1,votes[1:].sum(axis=1));pred=prob>=threshold
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    return dict(samples=len(y),positive_samples=int(np.sum(y==1)),negative_samples=int(np.sum(y==0)),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                true_positive=tp,false_positive=fp,false_negative=fn,threshold=threshold)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=HERE)
    p.add_argument('--samples',type=int,default=20000)
    p.add_argument('--trees',type=int,default=80)
    args=p.parse_args();root=args.root
    cv2.setNumThreads(2);cv2.setRNGSeed(42)
    pos=list((root/'game_data/sprites').glob('*.png'))
    neg=list((root/'game_data/non_projectiles').glob('*.png'))
    if len(pos)<10 or len(neg)<10:p.error('Run extract_assets.py first; need at least 10 source sprites per class')
    ptr,pte=split_sources(pos,42);ntr,nte=split_sources(neg,43)
    bg_paths=list((root/'learning/backgrounds').glob('*.png'))
    btr,bte=split_sources(bg_paths,44) if len(bg_paths)>1 else (bg_paths,[])
    load=lambda paths:[im for p in paths if (im:=read_image(p)) is not None]
    labels=[];groups={}
    for label,name in [(0,'other'),(1,'bullet')]:
        paths=list((root/'learning/labeled'/name).glob('*.png'))
        for path in paths:
            labels.append((label,path))
            sidecar=path.with_suffix('.png.json')
            group=json.loads(sidecar.read_text()).get('source_group',path.name) if sidecar.exists() else path.name
            groups.setdefault(group,[]).append((label,path))
    keys=sorted(groups);random.Random(45).shuffle(keys)
    cut=max(1,int(len(keys)*.8));ltr=[v for k in keys[:cut] for v in groups[k]];lte=[v for k in keys[cut:] for v in groups[k]]
    print(f'Building training set: {len(ptr)} positive source sprites, {len(ntr)} negative sources; real labels={len(labels)}',flush=True)
    x,y=build(ptr,ntr,ltr,load(btr),args.samples,46)
    # Calibration chooses threshold; final test uses different sprite identities.
    pcal,ptest=pte[::2],pte[1::2];ncal,ntest=nte[::2],nte[1::2]
    xc,yc=build(pcal,ncal,[],load(bte),2000,47)
    xt,yt=build(ptest,ntest,[],load(bte),2000,48)
    model=cv2.ml.RTrees_create();model.setMaxDepth(16);model.setMinSampleCount(4)
    model.setMaxCategories(2);model.setActiveVarCount(32)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,args.trees,0))
    print('Fitting classifier...',flush=True)
    start=time.perf_counter();model.train(x,cv2.ml.ROW_SAMPLE,y)
    calibration=[metrics(model,xc,yc,float(t)) for t in np.linspace(.35,.85,11)]
    eligible=[m for m in calibration if m['precision']>=.65] or calibration
    def f2(m):
        p,r=m['precision'],m['recall']
        return 5*p*r/max(1e-9,4*p+r)
    selected=max(eligible,key=f2)['threshold']
    report=dict(selected_threshold=selected,calibration=calibration,trained_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
                training_samples=len(y),training_source_sprites=len(ptr)+len(ntr),
                heldout_source_sprites=len(pte)+len(nte),human_labeled_patches=len(labels),human_source_groups=len(groups),
                synthetic_heldout=metrics(model,xt,yt,selected),
                scope='Patch classification on held-out composited sprites; NOT end-to-end gameplay detection or dodge success.')
    if lte:
        real_x=np.stack([features(read_image(path)) for _,path in lte])
        report['real_patch_heldout']=metrics(model,real_x,np.array([label for label,_ in lte]),selected)
    directory=root/'models';directory.mkdir(exist_ok=True)
    temp=directory/'projectile.pending.xml'
    fs=cv2.FileStorage('.xml',cv2.FILE_STORAGE_WRITE|cv2.FILE_STORAGE_MEMORY)
    model.write(fs,'opencv_ml_rtrees')
    temp.write_text(fs.releaseAndGetString(),encoding='utf-8')
    report_temp=directory/'report.pending.json'
    report_temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    os.replace(report_temp,directory/'report.json')
    os.replace(temp,directory/'projectile.xml')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

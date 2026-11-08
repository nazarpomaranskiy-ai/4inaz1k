"""One-class Fire Missile dataset: reviewed real frames + alpha-composited art.

Real holdout frames never become training backgrounds. Synthetic scores are
reported separately from real-game scores. No detector-generated labels are used.
"""
import argparse
import json
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image,ImageEnhance
from imaging import read_image,write_image

TRAIN_NEG=[0,50,100,150,200,250,900,1100]
VAL_NEG=[980,1200]
REAL_BOXES={1000:[(818,208,850,232),(780,480,807,509),(622,489,644,509)],
            1010:[(1054,65,1079,86),(959,681,984,710)],1030:[(830,417,858,439)]}

def write_labels(path,boxes,w,h):
    lines=[f'0 {(x1+x2)/2/w:.7f} {(y1+y2)/2/h:.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}' for x1,y1,x2,y2 in boxes]
    path.write_text('\n'.join(lines)+'\n' if lines else '',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-count',type=int,default=600);p.add_argument('--val-count',type=int,default=100)
    args=p.parse_args();rng=random.Random(5701);out=args.output
    rows=[json.loads(l) for l in (args.capture/'frames.jsonl').read_text().splitlines()]
    catalog=json.loads((args.project/'game_data/catalog.json').read_text('utf-8'))
    obj=next(o for o in catalog['objects'] if o['id']=='Fire Missile' and o['sprites'])
    sprite=Image.open(args.project/'game_data/sprites'/obj['sprites'][0]).convert('RGBA')
    backgrounds={i:Image.open(args.capture/rows[i]['image']).convert('RGB') for i in TRAIN_NEG+VAL_NEG}
    for split,count,ids in [('train',args.train_count,TRAIN_NEG),('val',args.val_count,VAL_NEG)]:
        images=out/'images'/split;labels=out/'labels'/split
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        for i in range(count):
            source=backgrounds[rng.choice(ids)];scale=rng.uniform(.8,1.4)
            bw,bh=max(640,round(source.width*scale)),max(640,round(source.height*scale))
            background=source.resize((bw,bh),Image.Resampling.BILINEAR)
            x=rng.randrange(bw-640+1);y=rng.randrange(bh-640+1)
            canvas=background.crop((x,y,x+640,y+640));boxes=[]
            for _ in range(0 if i%5==0 else rng.randint(2,8)):
                angle=rng.uniform(0,360);s=sprite.rotate(angle,Image.Resampling.NEAREST,expand=True)
                bbox=s.getbbox()
                if not bbox:continue
                s=s.crop(bbox);target=rng.randint(12,44);factor=target/max(s.size)
                s=s.resize((max(2,round(s.width*factor)),max(2,round(s.height*factor))),Image.Resampling.NEAREST)
                x=rng.randrange(640-s.width);y=rng.randrange(640-s.height)
                canvas.paste(s,(x,y),s);boxes.append((x,y,x+s.width,y+s.height))
            # Mild whole-image exposure shift; preserve the projectile's palette.
            if rng.random()<.4:canvas=ImageEnhance.Brightness(canvas).enhance(rng.uniform(.8,1.2))
            name=f'synthetic_{i:05d}'
            canvas.save(images/(name+'.jpg'),quality=rng.randint(78,96))
            write_labels(labels/(name+'.txt'),boxes,640,640)
        print(f'Generated {split}: {count} synthetic images',flush=True)
    # Reviewed actual images. Only frame 1000 is used as a real positive for training.
    for index in TRAIN_NEG+[1000]:
        image=Image.open(args.capture/rows[index]['image']).convert('RGB')
        for copy in range(12 if index==1000 else 3):
            name=f'real_{index:04d}_{copy:02d}';image.save(out/'images/train'/(name+'.jpg'),quality=95)
            write_labels(out/'labels/train'/(name+'.txt'),REAL_BOXES.get(index,[]),*image.size)
    real_holdout=[]
    for index in VAL_NEG+[1010,1030]:
        image=Image.open(args.capture/rows[index]['image']).convert('RGB');name=f'real_{index:04d}'
        image.save(out/'images/val'/(name+'.jpg'),quality=95)
        write_labels(out/'labels/val'/(name+'.txt'),REAL_BOXES.get(index,[]),*image.size)
        real_holdout.append(dict(image=str(out/'images/val'/(name+'.jpg')),index=index,boxes=REAL_BOXES.get(index,[])))
    (out/'dataset.yaml').write_text('path: '+out.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: Fire Missile\n',encoding='utf-8')
    manifest=dict(class_name='Fire Missile',source_capture=str(args.capture),source_sprite=obj['sprites'][0],
                  train_real_positive_frames=[1000],train_negative_frames=TRAIN_NEG,
                  real_holdout=real_holdout,synthetic_train=args.train_count,synthetic_val=args.val_count,
                  note='Real holdout uses different frames from one recording; limited one-class evidence, not production safety.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Dataset ready:',out,flush=True)

if __name__=='__main__':main()

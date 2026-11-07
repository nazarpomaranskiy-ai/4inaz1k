"""Human corrections for uncertain and missed objects; retrain local model."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tkinter as tk
from PIL import Image,ImageTk

ROOT=Path(__file__).resolve().parent

def store(image,label,source_group=None):
    folder=ROOT/'learning/labeled'/('bullet' if label else 'other')
    folder.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()
    image.save(folder/(digest+'.png'))
    (folder/(digest+'.png.json')).write_text(json.dumps(dict(label=label,source_group=source_group or digest,human_review=True)))

def main():
    root=tk.Tk();root.title('Teach detector: 1 bullet / 0 other / Space skip')
    files=sorted((ROOT/'learning/inbox').glob('*.png'));state={'index':0,'image':None,'photo':None}
    status=tk.StringVar();tk.Label(root,textvariable=status).pack()
    canvas=tk.Canvas(root,width=240,height=240,bg='#222222');canvas.pack()
    def show():
        canvas.delete('all')
        if state['index']>=len(files):
            status.set('No more uncertain patches. You can label missed bullets from saved frames.');return
        path=files[state['index']];image=Image.open(path).convert('RGB');state['image']=image
        state['photo']=ImageTk.PhotoImage(image.resize((240,240),Image.Resampling.NEAREST))
        canvas.create_image(0,0,anchor='nw',image=state['photo'])
        status.set(f'{state["index"]+1}/{len(files)}: 1 = bullet; 0 = other; Space = skip')
    def label(value):
        if state['index']>=len(files):return
        if value is not None:
            metadata=files[state['index']].with_suffix('.png.json')
            group=json.loads(metadata.read_text()).get('source_group') if metadata.exists() else None
            store(state['image'],value,group)
            files[state['index']].unlink(missing_ok=True)
            files[state['index']].with_suffix('.png.json').unlink(missing_ok=True)
        state['index']+=1;show()
    def annotate_frames():
        frames=sorted((ROOT/'learning/frames').glob('*.png'))
        if not frames:
            status.set('No saved frames yet. Run overlay with collect_learning enabled.');return
        win=tk.Toplevel(root);win.title('Drag a tight box around a missed bullet or a false detection')
        index=tk.IntVar(value=len(frames)-1);mode=tk.IntVar(value=1)
        header=tk.Frame(win);header.pack()
        tk.Radiobutton(header,text='Bullet',variable=mode,value=1).pack(side='left')
        tk.Radiobutton(header,text='Other / false detection',variable=mode,value=0).pack(side='left')
        c=tk.Canvas(win);c.pack();view={'image':None,'photo':None,'scale':1,'start':None}
        def display():
            im=Image.open(frames[index.get()]).convert('RGB');scale=min(1,1100/im.width,700/im.height)
            view.update(image=im,scale=scale)
            view['photo']=ImageTk.PhotoImage(im.resize((round(im.width*scale),round(im.height*scale))))
            c.configure(width=round(im.width*scale),height=round(im.height*scale))
            c.delete('all');c.create_image(0,0,anchor='nw',image=view['photo'])
            win.title(f'Frame {index.get()+1}/{len(frames)}: drag object box; label immediately saved')
        def next_frame(delta):
            index.set(max(0,min(len(frames)-1,index.get()+delta)));display()
        tk.Button(header,text='Previous frame',command=lambda:next_frame(-1)).pack(side='left')
        tk.Button(header,text='Next frame',command=lambda:next_frame(1)).pack(side='left')
        def down(e):view['start']=(e.x,e.y)
        def drag(e):
            if view['start']:
                c.delete('box');c.create_rectangle(*view['start'],e.x,e.y,outline='yellow',tags='box')
        def up(e):
            if not view['start']:return
            sx,sy=view['start'];scale=view['scale'];view['start']=None
            l,r=sorted([round(sx/scale),round(e.x/scale)]);t,b=sorted([round(sy/scale),round(e.y/scale)])
            l,t=max(0,l),max(0,t);r,b=min(view['image'].width,r),min(view['image'].height,b)
            if r-l>=3 and b-t>=3:
                store(view['image'].crop((l,t,r,b)),mode.get(),hashlib.sha256(frames[index.get()].read_bytes()).hexdigest());c.itemconfigure('box',outline='#00ff88')
        c.bind('<ButtonPress-1>',down);c.bind('<B1-Motion>',drag);c.bind('<ButtonRelease-1>',up);display()
    def train():
        directory=ROOT/'learning';directory.mkdir(exist_ok=True)
        log=(directory/'training.log').open('w',encoding='utf-8')
        process=subprocess.Popen([sys.executable,str(ROOT/'train_detector.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        log.close();status.set('Training started. Model reloads in overlay when complete.')
        def check():
            code=process.poll()
            if code is None:root.after(500,check)
            else:status.set('Training completed; model reloaded.' if code==0 else 'Training failed: see learning/training.log')
        root.after(500,check)
    buttons=tk.Frame(root);buttons.pack()
    for text,value in [('Bullet (1)',1),('Other (0)',0),('Skip',None)]:
        tk.Button(buttons,text=text,command=lambda v=value:label(v)).pack(side='left')
    tk.Button(root,text='Teach missed bullets from frames',command=annotate_frames).pack(fill='x')
    tk.Button(root,text='Train with corrections',command=train).pack(fill='x')
    root.bind('1',lambda e:label(1));root.bind('0',lambda e:label(0));root.bind('<space>',lambda e:label(None))
    show();root.mainloop()

if __name__=='__main__':main()

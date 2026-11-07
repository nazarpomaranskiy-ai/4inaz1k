"""Read an installed Unity resources.assets and export local projectile knowledge.

No game file writes. Generated art/data stays in game_data/, excluded from Git.
FlatBuffer field layout follows publicly documented SpriteSheetRoot/SpriteSheet/
Sprite/Position types; no client code is loaded or executed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import struct
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

class FlatReader:
    def __init__(self,data): self.data=data
    def unpack(self,fmt,pos): return struct.unpack_from('<'+fmt,self.data,pos)[0]
    def slot(self,table,index):
        vt=table-self.unpack('i',table)
        offset=4+index*2
        if offset>=self.unpack('H',vt):return None
        value=self.unpack('H',vt+offset)
        return table+value if value else None
    def scalar(self,table,index,fmt,default=0):
        p=self.slot(table,index)
        return self.unpack(fmt,p) if p is not None else default
    def target(self,pos):return pos+self.unpack('I',pos)
    def string(self,table,index,default=''):
        p=self.slot(table,index)
        if p is None:return default
        s=self.target(p); n=self.unpack('I',s)
        return self.data[s+4:s+4+n].decode('utf-8')
    def vector(self,table,index):
        p=self.slot(table,index)
        if p is None:return []
        v=self.target(p); n=self.unpack('I',v)
        if n>200000:raise ValueError('Unexpected sprite vector size')
        return [self.target(v+4+i*4) for i in range(n)]
    def sprites(self):
        root=self.unpack('I',0)
        for sheet in self.vector(root,0):
            name=self.string(sheet,0); atlas=self.scalar(sheet,1,'q')
            for sprite in self.vector(sheet,2):
                pos=self.slot(sprite,0)
                if pos is None:continue
                x,y,h,w=struct.unpack_from('<ffff',self.data,pos)
                yield (name,self.scalar(sprite,3,'i')),dict(atlas=self.scalar(sprite,7,'q',atlas),
                      box=[round(x),round(y),round(x+w),round(y+h)])

def number(text):
    text=(text or '0').strip()
    return int(text,16) if text.lower().startswith('0x') else int(text)

def parse_xml(paths):
    objects=[]; attacks=[]; failures=[]
    for path in sorted(paths):
        try:root=ET.parse(path).getroot()
        except ET.ParseError as exc:
            failures.append(dict(file=path.name,error=str(exc)));continue
        for obj in root.iter('Object'):
            identity=dict(id=obj.get('id',''),type=obj.get('type',''),source=path.name)
            if obj.findtext('Class')=='Projectile':
                textures=[]
                for t in obj.iter('Texture'):
                    if t.find('File') is not None and t.find('Index') is not None:
                        textures.append(dict(sheet=t.findtext('File'),index=number(t.findtext('Index'))))
                objects.append(identity|dict(textures=textures,definition=ET.tostring(obj,encoding='unicode')))
            for shot in obj.findall('Projectile'):
                attacks.append(identity|dict(slot=shot.get('id'),projectile=shot.findtext('ObjectId'),
                       fields={x.tag:x.text.strip() if x.text else True for x in shot},
                       definition=ET.tostring(shot,encoding='unicode')))
    return objects,attacks,failures

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('game_data'))
    args=p.parse_args()
    import UnityPy
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    xml=out/'xml';xml.mkdir(exist_ok=True)
    print('Reading local assets (no game files are modified)...',flush=True)
    env=UnityPy.load(str(args.resources)); atlases={}; blob=None; xml_names={}
    for o in env.objects:
        if o.type.name=='TextAsset':
            d=o.read(); raw=d.m_Script.encode('utf-8','surrogateescape') if isinstance(d.m_Script,str) else d.m_Script
            if raw.lstrip().startswith(b'<'):
                filename=f'{o.path_id}.xml';(xml/filename).write_bytes(raw);xml_names[filename]=d.m_Name
            elif d.m_Name=='spritesheetf':blob=raw
        elif o.type.name=='Texture2D':
            d=o.read()
            if d.m_Name in ('characters','mapObjects'):
                atlases[2 if d.m_Name=='characters' else 4]=d.image.convert('RGBA')
    objects,attacks,failures=parse_xml(xml.glob('*.xml'))
    lookup=dict(FlatReader(blob).sprites()) if blob else {}
    sprite_dir=out/'sprites';sprite_dir.mkdir(exist_ok=True)
    sprites={};missing=[]
    for obj in objects:
        obj['sprites']=[]
        for texture in obj['textures']:
            key=(texture['sheet'],texture['index']); location=lookup.get(key)
            if not location or location['atlas'] not in atlases:
                missing.append(dict(object=obj['id'],texture=texture));continue
            image=atlases[location['atlas']].crop(location['box'])
            if image.width<1 or image.height<1 or image.width>256 or image.height>256:continue
            bbox=image.getbbox()
            if not bbox:continue
            image=image.crop(bbox)
            digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()[:24]
            filename=digest+'.png'
            if filename not in sprites:
                image.save(sprite_dir/filename)
                sprites[filename]=dict(file=filename,width=image.width,height=image.height,
                                       sheet=texture['sheet'],index=texture['index'])
            obj['sprites'].append(filename)
    projectile_keys={(t['sheet'],t['index']) for obj in objects for t in obj['textures']}
    negative_keys=set()
    for path in xml.glob('*.xml'):
        try:root=ET.parse(path).getroot()
        except ET.ParseError:continue
        for obj in root.iter('Object'):
            if obj.findtext('Class')=='Projectile':continue
            for texture in obj.iter('Texture'):
                if texture.find('File') is not None and texture.find('Index') is not None:
                    key=(texture.findtext('File'),number(texture.findtext('Index')))
                    if key not in projectile_keys:negative_keys.add(key)
    choices=sorted(negative_keys);random.Random(42).shuffle(choices)
    negative_dir=out/'non_projectiles';negative_dir.mkdir(exist_ok=True)
    negatives=0
    for key in choices:
        if negatives>=1500:break
        location=lookup.get(key)
        if not location or location['atlas'] not in atlases:continue
        image=atlases[location['atlas']].crop(location['box'])
        bbox=image.getbbox()
        if not bbox or image.width>128 or image.height>128:continue
        image=image.crop(bbox)
        digest=hashlib.sha256(image.tobytes()+str(image.size).encode()).hexdigest()[:24]
        image.save(negative_dir/(digest+'.png'));negatives+=1
    catalog=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),
       source='Read-only extraction from user-installed RotMG Exalt resources.assets',
       resource_bytes=args.resources.stat().st_size,xml_names=xml_names,objects=objects,
       attacks=attacks,sprites=list(sprites.values()),unresolved_textures=missing,xml_errors=failures,
       coverage_note='All parsed Projectile objects in this local file; not a claim of all live server attacks or perfect visual recognition.')
    (out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(xml_files=len(xml_names),projectile_objects=len(objects),attack_definitions=len(attacks),
                unique_sprite_images=len(sprites),negative_sprite_examples=negatives,unresolved_texture_references=len(missing),xml_errors=len(failures))
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()

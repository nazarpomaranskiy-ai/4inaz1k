import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from imaging import read_image,write_image
from learning import PatchModel,VisionEngine,features
from extract_assets import parse_xml

class LearningTests(unittest.TestCase):
    def test_unicode_image_paths_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'приклад.png';image=np.full((12,12,3),90,np.uint8)
            write_image(path,image);np.testing.assert_array_equal(read_image(path),image)

    def test_catalog_preserves_motion_parameters(self):
        xml='<Objects><Object id="Enemy"><Projectile id="1"><ObjectId>Shot</ObjectId><Speed>120</Speed><Wavy/></Projectile></Object><Object id="Shot"><Class>Projectile</Class><Texture><File>sheet</File><Index>0x10</Index></Texture></Object></Objects>'
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sample.xml';p.write_text(xml)
            objects,attacks,failures=parse_xml([p])
            self.assertEqual(objects[0]['textures'][0]['index'],16)
            self.assertEqual(attacks[0]['fields']['Wavy'],True)
            self.assertEqual(attacks[0]['projectile'],'Shot')
            self.assertEqual(failures,[])

    def test_proposals_include_green_white_and_red(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(player=[100,100],collect_learning=False)
            engine=VisionEngine(cfg,folder)
            engine.model.probabilities=lambda patches:np.full(len(patches),.9)
            frame=np.zeros((200,300,3),np.uint8)
            for x,color in [(40,(0,255,0)),(120,(255,255,255)),(220,(0,0,255))]:
                cv2.circle(frame,(x,100),5,color,-1)
            found,_=engine.detect(frame)
            for expected in [40,120,220]:
                self.assertTrue(any(abs(x-expected)<3 and abs(y-100)<3 for x,y,r in found),expected)
            engine.close()

    def test_model_memory_serialization_with_unicode_path(self):
        rng=np.random.default_rng(42)
        images=[];labels=[]
        for i in range(30):
            image=rng.integers(0,30,(24,24,3),dtype=np.uint8)
            if i%2:cv2.circle(image,(12,12),6,(255,255,255),-1)
            images.append(image);labels.append(i%2)
        model=cv2.ml.RTrees_create();model.setMaxDepth(8)
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,12,0))
        model.train(np.stack([features(i) for i in images]),cv2.ml.ROW_SAMPLE,np.array(labels,np.int32))
        fs=cv2.FileStorage('.xml',cv2.FILE_STORAGE_WRITE|cv2.FILE_STORAGE_MEMORY)
        model.write(fs,'opencv_ml_rtrees')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'модель.xml';path.write_text(fs.releaseAndGetString(),encoding='utf-8')
            loaded=PatchModel(path);prob=loaded.probabilities(images)
            self.assertGreater(float(prob[1::2].mean()),.7)
            self.assertLess(float(prob[::2].mean()),.3)

    def test_reviewed_negative_memory_overrides_false_positive(self):
        from types import SimpleNamespace
        image=np.full((24,24,3),90,np.uint8)
        cv2.circle(image,(12,12),5,(0,0,255),-1)
        value=features(image);value/=np.linalg.norm(value)
        model=PatchModel.__new__(PatchModel)
        model.reload=lambda:None
        model.model=SimpleNamespace(getVotes=lambda values,flags:np.array([[0,1],[0,10]],np.int32))
        model.memory={0:value[None,:]}
        self.assertEqual(float(model.probabilities([image])[0]),0)

    def test_conflicting_corrections_do_not_force_a_label(self):
        from types import SimpleNamespace
        image=np.full((24,24,3),90,np.uint8)
        value=features(image);value/=np.linalg.norm(value)
        model=PatchModel.__new__(PatchModel);model.reload=lambda:None
        model.model=SimpleNamespace(getVotes=lambda values,flags:np.array([[0,1],[4,6]],np.int32))
        model.memory={0:value[None,:],1:value[None,:]}
        self.assertAlmostEqual(float(model.probabilities([image])[0]),.6)

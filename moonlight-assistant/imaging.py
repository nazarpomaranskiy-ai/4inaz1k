"""OpenCV image IO via Python paths (including Cyrillic Windows folders)."""
from pathlib import Path
import cv2
import numpy as np

def read_image(path,flags=cv2.IMREAD_COLOR):
    try:return cv2.imdecode(np.frombuffer(Path(path).read_bytes(),np.uint8),flags)
    except (OSError,cv2.error):return None

def write_image(path,image):
    path=Path(path)
    ok,data=cv2.imencode(path.suffix or '.png',image)
    if not ok:raise OSError(f'Could not encode image: {path}')
    path.write_bytes(data.tobytes())

"""Start the experimental advisory overlay using a locally configured environment."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    runtime=ROOT/'neural_runtime.json'
    if not runtime.exists():
        raise SystemExit('Missing neural_runtime.json: configure the Python environment with CUDA PyTorch and ultralytics first.')
    python=Path(json.loads(runtime.read_text('utf-8'))['python'])
    if not python.is_file():raise SystemExit(f'Configured Python does not exist: {python}')
    return subprocess.call([str(python),str(ROOT/'overlay.py'),'--neural',*sys.argv[1:]],cwd=ROOT)

if __name__=='__main__':raise SystemExit(main())

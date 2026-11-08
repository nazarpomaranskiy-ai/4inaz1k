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
    python=Path(json.loads(runtime.read_text('utf-8-sig'))['python'])
    if not python.is_file():raise SystemExit(f'Configured Python does not exist: {python}')
    args=sys.argv[1:]
    if '--observe' in args:
        args.remove('--observe')
        command=[str(python),str(ROOT/'observe_game.py'),*args]
    else:command=[str(python),str(ROOT/'overlay.py'),'--neural',*args]
    return subprocess.call(command,cwd=ROOT)

if __name__=='__main__':raise SystemExit(main())

"""Run the recorded C1 training/replay with NumPy-safe JSON serialization."""
import json,runpy
from pathlib import Path
import run_experiments as driver
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
 temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n');temporary.replace(path)
driver.save=save
runpy.run_path(str(Path(__file__).resolve().parent/'run_comment1.py'),run_name='__main__')
runpy.run_path(str(Path(__file__).resolve().parent/'finish_replay.py'),run_name='__main__')

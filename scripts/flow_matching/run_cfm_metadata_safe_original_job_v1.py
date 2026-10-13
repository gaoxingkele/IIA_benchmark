"""Original numerical job, with bounded retries for Windows metadata replaces.

No fitting, sampling, seed, solver or update-budget code changes. Only JSON
publication uses a unique temporary file and retries transient PermissionError.
CFM-TS: https://openreview.net/forum?id=Hqn4Aj7xrQ.
"""
import json
from pathlib import Path
import os
import time
from types import FunctionType
import uuid


def safe_write(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        temporary.write_text(content,encoding='utf-8')
        for attempt in range(8):
            try:
                os.replace(temporary,path)
                return
            except PermissionError:
                if attempt==7:raise
                time.sleep(.025*2**attempt)
    finally:
        if temporary.exists():temporary.unlink()


def namespace():
    from scripts.flow_matching import run_cfm_ts_original_job as original
    globals()['original']=original
    values=dict(original.__dict__,write=safe_write)
    for name,fn in original.__dict__.items():
        if isinstance(fn,FunctionType) and fn.__module__==original.__name__:
            values[name]=FunctionType(fn.__code__,values,fn.__name__,fn.__defaults__,fn.__closure__)
    values['write']=safe_write
    return values


def main():namespace()['main']()


if __name__=='__main__':main()

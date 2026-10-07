from __future__ import annotations
import csv, hashlib, json, os, platform, random, subprocess, time
from pathlib import Path
import numpy as np

def seed_for(*parts):
    raw = json.dumps(["STE-RunPod-Genuine-v1", *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)

def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

def emit(**items):
    value={"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **items}
    print(json.dumps(value), flush=True)
    if os.environ.get("STE_PROGRESS_FILE"): atomic_json(os.environ["STE_PROGRESS_FILE"],value)

def write_rows(path, rows):
    if not rows: raise ValueError("Refusing an empty result table: " + str(path))
    import pandas as pd
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

def configure_torch(seed, device):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)

def environment(device, smoke):
    import torch, scipy, pandas
    try: smi=subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError): smi="unavailable"
    return {"python":platform.python_version(), "platform":platform.platform(),
            "torch":torch.__version__, "numpy":np.__version__, "scipy":scipy.__version__,
            "pandas":pandas.__version__, "cuda_runtime":torch.version.cuda,
            "device":str(device), "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            "nvidia_smi":smi, "smoke":smoke,
            "torch_threads":torch.get_num_threads(), "deterministic_algorithms":True,
            "tf32":False, "mixed_precision":False}

def finish_unit(path, required):
    path=Path(path)
    hashes={str(Path(f).relative_to(path)):sha(f) for f in required}
    atomic_json(path/"COMPLETE.json", {"files":hashes})

def unit_complete(path):
    path=Path(path); mark=path/"COMPLETE.json"
    if not mark.exists(): return False
    data=json.loads(mark.read_text())
    for rel, digest in data["files"].items():
        if not (path/rel).is_file() or sha(path/rel)!=digest:
            raise RuntimeError("Completed artifact changed or missing: " + str(path/rel))
        if Path(rel).name=="COMPLETE.json":
            if not unit_complete((path/rel).parent): raise RuntimeError("Nested completed unit failed verification")
    return True

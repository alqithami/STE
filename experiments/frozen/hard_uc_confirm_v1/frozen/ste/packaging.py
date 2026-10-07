from pathlib import Path
import hashlib,json,os,zipfile
from .common import atomic_json,sha

def source_files(code):
    return [p for p in sorted(Path(code).rglob("*")) if p.is_file() and not any(x in p.parts for x in ("__pycache__",".venv")) and p.suffix not in (".pyc",)]

def create_archive(path,code,run,classification,review=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); items=[]
    for p in source_files(code): items.append((p,"code/"+str(p.relative_to(code))))
    for p in sorted(Path(run).rglob("*")):
        if not p.is_file() or p.name in ("RUN.lock","ARCHIVES_READY.json") or p.suffix in (".tmp",): continue
        if review:
            if p.suffix==".pt" or "datasets" in p.parts: continue
            if p.name in ("observations.npz","posterior_joint_samples.npz"): continue
            if "counts" in p.parts and p.name=="scores.npz": continue
        items.append((p,"run/"+str(p.relative_to(run))))
    manifest={"classification":classification,"review":review,"files":[]}
    tmp=path.with_suffix(path.suffix+".tmp")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p,name in items:
            row={"path":name,"bytes":p.stat().st_size,"sha256":sha(p)}
            z.write(p,name); manifest["files"].append(row)
        z.writestr("ARCHIVE_MANIFEST.json",json.dumps(manifest,indent=2,sort_keys=True))
    os.replace(tmp,path)
    from verify_archive import verify
    checked=verify(path); digest=sha(path)
    path.with_suffix(path.suffix+".sha256").write_text(digest+"  "+path.name+"\n")
    return {**checked,"sha256":digest}

def package_success(code,cfg,run):
    run=Path(run); prefix="STE_RUNPOD_V1_SMOKE" if cfg.get("smoke") else "STE_RUNPOD_V1"
    classification="software smoke; not scientific evidence" if cfg.get("smoke") else "fresh experiment; results require scientific interpretation"
    full=create_archive(run.parent/(prefix+"_FULL.zip"),code,run,classification,False)
    review=create_archive(run.parent/(prefix+"_REVIEW.zip"),code,run,classification,True)
    atomic_json(run/"ARCHIVES_READY.json",{"status":"SMOKE_ARCHIVES_READY" if cfg.get("smoke") else "STE_RUNPOD_V1_ARCHIVES_READY","full":full,"review":review})
    return full,review

def package_failure(code,run):
    return create_archive(Path(run).parent/"STE_RUNPOD_V1_FAILURE.zip",code,run,"incomplete/failed; not completed evidence",True)

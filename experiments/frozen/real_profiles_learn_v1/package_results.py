#!/usr/bin/env python3
"""Verify complete results, then create small and full inspectable archives."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results');args=ap.parse_args();out=args.out.resolve()
    subprocess.run([sys.executable,str(ROOT/'run.py'),'--out',str(out),'--stage','verify'],check=True)
    subprocess.run([sys.executable,str(ROOT/'audit_results.py'),'--out',str(out)],check=True)
    if not json.loads((out/'COMPLETE.json').read_text())['production_results']:
        stem='STE_RealProfiles_Learn_v1_SMOKE'
    else:stem='STE_RealProfiles_Learn_v1'
    payload=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and 'results' not in p.relative_to(ROOT).parts
             and not any(x.startswith('.') or x=='__pycache__' for x in p.relative_to(ROOT).parts)
             and p.suffix not in ('.zip','.sha256','.pyc')]
    payload+=[p for p in sorted(out.rglob('*')) if p.is_file() and p.suffix!='.tmp' and p.name!='RUN.log']
    mapped={}
    for p in payload:
        rel='results/'+str(p.relative_to(out)) if p.is_relative_to(out) else str(p.relative_to(ROOT));mapped[rel]=p
    records={name:{'sha256':digest(p),'bytes':p.stat().st_size} for name,p in mapped.items()}
    result=[]
    for mode in ('REVIEW','FULL'):
        path=ROOT/(stem+'_'+mode+'.zip');selected={k:p for k,p in mapped.items() if mode=='FULL' or p.suffix!='.pt'}
        manifest={name:records[name] for name in selected}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,p in selected.items():z.write(p,'STE_RealProfiles_Learn_v1/'+name)
            z.writestr('STE_RealProfiles_Learn_v1/ARCHIVE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise RuntimeError('Archive CRC failed')
            for name,record in manifest.items():
                if hashlib.sha256(z.read('STE_RealProfiles_Learn_v1/'+name)).hexdigest()!=record['sha256']:raise RuntimeError('Archive member hash failed')
        path.with_suffix('.zip.sha256').write_text(digest(path)+'  '+path.name+'\n')
        result.append({'archive':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'members_verified':len(manifest)})
    print(json.dumps({'archives':result},indent=2),flush=True)
    print('REAL_LEARNING_ARCHIVES_READY',flush=True)

if __name__=='__main__':main()

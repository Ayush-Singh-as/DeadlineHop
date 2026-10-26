"""Create a clean conventional Git checkout and ZIP without local runtime caches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='dist/radio-rl')
    args=p.parse_args()
    dest=Path(args.output).resolve()
    if dest.exists(): raise FileExistsError(dest)
    gitdir=ROOT/'.development-history'
    if not gitdir.exists(): gitdir=ROOT/'.git'
    subprocess.run(['git','clone','--no-hardlinks',str(gitdir),str(dest)],check=True)
    # Cloning a local gitdir can preserve an absolute core.worktree setting only
    # in the source; the clone is independent and uses conventional metadata.
    subprocess.run(['git','-C',str(dest),'remote','remove','origin'],check=True)
    entries={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(dest.rglob('*')) if p.is_file() and '.git' not in p.relative_to(dest).parts}
    (dest/'RELEASE_MANIFEST.json').write_text(json.dumps(entries,indent=2))
    archive=dest.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(dest.rglob('*')):
            if path.is_file(): z.write(path,path.relative_to(dest.parent))
    print(json.dumps(dict(checkout=str(dest),archive=str(archive),files=len(entries)),indent=2))


if __name__=='__main__': main()

"""Crop a prepared case without interpolation and regenerate its input manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from results import config


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
    return digest.hexdigest()


def crop(source, output, bounds):
    if output.exists(): raise ValueError('Output already exists')
    if (source/'INCOMPLETE').exists(): raise ValueError('Source conversion incomplete')
    expected={'case.cfg','u.bin','v.bin','w.bin','solid.bin'}
    seen=set()
    for line in (source/'input.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        if name not in expected or name in seen: raise ValueError('Unexpected or duplicate manifest entry')
        seen.add(name)
        if sha(source/name)!=digest: raise ValueError(f'Checksum failed: {name}')
    if seen != expected: raise ValueError('Incomplete input manifest')
    if len(bounds)!=3 or any(not np.isfinite([lo,hi]).all() or lo>hi for lo,hi in bounds):
        raise ValueError('Invalid crop bounds')
    cfg=config(source/'case.cfg')
    dimensions=[int(cfg[f'n{d}']) for d in 'xyz']
    slices=[]; new={}
    for d,n,(lo,hi) in zip('xyz',dimensions,bounds):
        spacing=float(cfg[f'd{d}']);origin=float(cfg[f'origin_{d}'])
        centres=origin+(np.arange(n)+.5)*spacing
        keep=np.flatnonzero((centres>=lo)&(centres<=hi))
        if not len(keep): raise ValueError(f'Empty {d} crop')
        start,stop=int(keep[0]),int(keep[-1])+1
        slices.append(slice(start,stop))
        new.update({f'n{d}':stop-start,f'd{d}':spacing,f'origin_{d}':origin+start*spacing})
    output.mkdir(parents=True)
    (output/'INCOMPLETE').write_text('Crop in progress\n')
    sx,sy,sz=slices
    for name,dtype in [('u.bin','<f8'),('v.bin','<f8'),('w.bin','<f8'),('solid.bin','u1')]:
        data=np.memmap(source/name,dtype=dtype,mode='r',shape=tuple(dimensions[::-1]))
        with (output/name).open('wb') as stream:
            for k in range(sz.start,sz.stop):
                np.ascontiguousarray(data[k,sy,sx]).tofile(stream)
    (output/'case.cfg').write_text(''.join(f'{k}={v}\n' for k,v in new.items()))
    provenance=dict(source_case=str(source.resolve()),source_manifest=(source/'input.sha256').read_text(),
                    requested_centre_bounds_xyz=bounds,index_ranges_xyz=[[s.start,s.stop] for s in slices],
                    geometry=new,modelling_notes=[
                    'Interior rectangular subdomain; no interpolation or wind correction.',
                    'The new horizontal lower face is a simplified floor, not reconstructed terrain.',
                    'Building-only STL mask retained; wind component mapping remains provisional.',
                    'Use as a numerical pilot, not a validated neighbourhood forecast.'])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'input.sha256').write_text(''.join(f'{sha(output/name)}  {name}\n' for name in ['case.cfg','u.bin','v.bin','w.bin','solid.bin']))
    (output/'INCOMPLETE').unlink()
    return provenance


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--bounds',nargs=6,type=float,required=True,metavar=('XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'))
    a=p.parse_args()
    print(json.dumps(crop(a.source,a.output,list(zip(a.bounds[::2],a.bounds[1::2]))),indent=2))

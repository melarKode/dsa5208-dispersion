"""Fetch only STL members from a range-enabled ZIP, checking ZIP CRCs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = 'https://zenodo.org/records/21674773/files/cell_centers.zip?download=1'


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.transferred = url, 0, 0
        with self.request(0, 0) as response:
            header = response.headers.get('Content-Range', '')
            self.size = int(header.rsplit('/', 1)[1])
            response.read(1)

    def request(self, first, last):
        req = urllib.request.Request(self.url, headers={
            'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity',
            'User-Agent': 'dispersion-geometry-inspection/1.0'})
        response = urllib.request.urlopen(req, timeout=90)
        expected = f'bytes {first}-{last}/'
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
            response.close()
            raise RuntimeError('Server did not honor byte range; stopped without downloading full archive. Use --archive with a cluster-local ZIP instead.')
        return response

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if target < 0:
            raise ValueError('Negative ZIP seek')
        self.position = target
        return target

    def read(self, n=-1):
        n = self.size-self.position if n < 0 else min(n,self.size-self.position)
        if n <= 0:
            return b''
        if n > 32*1024**2:
            raise RuntimeError('Unexpected ZIP read over 32 MiB')
        if self.transferred+n > 512*1024**2:
            raise RuntimeError('Partial-download budget exceeded (512 MiB)')
        with self.request(self.position,self.position+n-1) as response:
            data=response.read(n+1)
        if len(data) != n:
            raise RuntimeError('Incomplete range response')
        self.position += n
        self.transferred += n
        return data


def extract(source, output):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        inventory=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc32=f'{i.CRC:08x}') for i in archive.infolist()]
        (output/'archive-members.json').write_text(json.dumps(inventory,indent=2)+'\n')
        selected=[i for i in archive.infolist() if i.filename.lower().endswith('.stl')]
        if not selected:
            raise RuntimeError('No STL found; inspect archive-members.json')
        if sum(i.file_size for i in selected) > 512*1024**2:
            raise RuntimeError('STL extraction exceeds 512 MiB; inspect inventory first')
        manifest=[]
        for number, member in enumerate(selected):
            destination=output/f'{number:02d}-{Path(member.filename).name}'
            if destination.exists():
                raise RuntimeError(f'Refusing to overwrite {destination}')
            temporary=destination.with_suffix('.stl.partial')
            with archive.open(member) as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp,out,length=1024**2)
            temporary.rename(destination)
            digest=hashlib.sha256(destination.read_bytes()).hexdigest()
            manifest.append(dict(archive_member=member.filename,local_file=destination.name,sha256=digest,crc32=f'{member.CRC:08x}',bytes=member.file_size))
            print(f'Extracted and CRC-verified {destination}',flush=True)
    (output/'geometry-source.json').write_text(json.dumps(dict(url=URL,members=manifest),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('data/geometry'))
    p.add_argument('--archive',type=Path)
    a=p.parse_args()
    source=a.archive if a.archive else RemoteZip(URL)
    extract(source,a.output)
    if isinstance(source,RemoteZip):
        print(f'ZIP range bytes transferred: {source.transferred}')

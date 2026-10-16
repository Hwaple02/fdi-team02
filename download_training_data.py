"""Download the private Supabase ZIP using a temporary signed URL."""
import argparse
import getpass
import hashlib
import io
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from load_training_data import PERIODS, VARIABLES


def download_training_data(signed_url, destination, manifest_path=None):
    parsed = urllib.parse.urlsplit(signed_url)
    if parsed.scheme != 'https' or parsed.hostname != 'jiruebyoipsuzywedbhq.supabase.co':
        raise ValueError('이 프로젝트의 HTTPS Supabase 링크만 허용합니다.')
    manifest_path = manifest_path or Path(__file__).with_name('data_manifest.json')
    manifest = json.loads(Path(manifest_path).read_text())
    with urllib.request.urlopen(signed_url, timeout=60) as response:
        data = response.read(manifest['bytes'] + 1)
    if len(data) != manifest['bytes'] or hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise ValueError('데이터 크기 또는 SHA-256 불일치')
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    allowed = {f'raw/{period}__{v}.csv' for period in PERIODS for v in VARIABLES}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != allowed | {'README.txt'}:
            raise ValueError('예상하지 않은 압축파일 구성')
        for name in sorted(allowed):
            target = dest / Path(name).name
            content = archive.read(name)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'기존 데이터 불일치: {target.name}')
        for name in sorted(allowed):
            (dest / Path(name).name).write_bytes(archive.read(name))
    return dest


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    url=getpass.getpass('Supabase 서명 다운로드 URL (입력 숨김): ')
    download_training_data(url,args.output_dir)
    print('학습 데이터 12개 다운로드 및 해시 검사 완료')

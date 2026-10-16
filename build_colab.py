"""Regenerate the standalone notebook from reviewed repository sources."""
import json
from pathlib import Path

root = Path(__file__).parent
cells=[]
def markdown(text):
    cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
def code(text):
    cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(True),'outputs':[],'execution_count':None})

markdown('''# Team 02 · FM 최소 프로토타입
규모·모멘텀 두 변수로 내부 검증을 실행합니다. 최종 연구 모형은 아닙니다.
모멘텀: 입력월 t−12~t−2(11개월). 테스트 데이터는 사용하지 않습니다.
Supabase 서명 URL은 팀 내부에서 전달받고, 공개 노트북 셀에 적지 마세요.
셀을 위에서 아래로 실행합니다. 현재 런타임에서 이미 다른 pandas/numpy를 사용했다면 패키지 설치 후 런타임을 다시 시작하세요.
''')
code('%pip install -q pandas==2.2.3 numpy==2.3.5\n')
sources={name:(root/name).read_text() for name in ['load_training_data.py','fm_pipeline.py','run_all.py','download_training_data.py','config.json','data_manifest.json']}
code('from pathlib import Path\nimport os\nWORK=Path("/content/fdi_team02")\nWORK.mkdir(exist_ok=True)\nos.chdir(WORK)\nSOURCES='+repr(sources)+'\nfor name,content in SOURCES.items():\n    (WORK/name).write_text(content,encoding="utf-8")\n')
markdown('## 비공개 학습 자료 다운로드\nColab Secret `FDI_TRAINING_URL` 또는 숨김 입력을 사용합니다. 링크가 만료되면 프로젝트 소유자에게 새 서명 URL을 받으세요.')
code('''import getpass
from download_training_data import download_training_data
try:
    from google.colab import userdata
    signed_url=userdata.get('FDI_TRAINING_URL')
except Exception:
    signed_url=getpass.getpass('Supabase 서명 URL: ')
download_training_data(signed_url, WORK/'data/raw')
del signed_url
print('학습 자료 12개 다운로드·해시 확인 완료')
''')
code('''import subprocess,sys,datetime
OUTPUT=WORK/'artifacts'/datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
subprocess.run([sys.executable,'run_all.py','--data-dir',str(WORK/'data/raw'),'--output-dir',str(OUTPUT)],check=True)
''')
code('''import json,pandas as pd
summary=json.loads((OUTPUT/'validation_summary.json').read_text())
display(pd.DataFrame(summary['folds']))
print('108개월 내부 검증 평균 IC:',summary['overall_mean_ic'])
print('산출물 폴더:',OUTPUT)
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'colab_prototype.ipynb'}},'nbformat':4,'nbformat_minor':5}
(root/'colab_prototype.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))

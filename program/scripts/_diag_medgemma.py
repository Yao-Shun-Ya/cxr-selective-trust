import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageFilter
from src.cxr_trust.models.medgemma import MedGemma
from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt

model_path = Path('models/medgemma/MedGemma-4B-IT')
img_path = Path('data/nih/images/images/00028873_008.png')
orig = Image.open(img_path).convert('RGB')
blank = Image.new('RGB', orig.size, (128,128,128))
blur = orig.filter(ImageFilter.GaussianBlur(radius=8))
out_dir = Path('outputs/diagnose_medgemma'); out_dir.mkdir(parents=True, exist_ok=True)
blank_path=out_dir/'blank.png'; blur_path=out_dir/'blur.png'
blank.save(blank_path); blur.save(blur_path)
print('loading...')
model = MedGemma(str(model_path), device_map='auto', load_in_4bit=True, max_memory={0:'7GiB','cpu':'25GiB'})
model.load()
prompt = SIMPLE_SYSTEM_PROMPT + chr(10)*2 + build_simple_prompt()
for name, p in [('original', img_path), ('blank', blank_path), ('blur', blur_path)]:
    raw=model.generate(p, prompt, max_new_tokens=96)
    print('\n===',name,'===')
    print(raw[:1500])
    (out_dir/f'{name}.txt').write_text(raw,encoding='utf-8')

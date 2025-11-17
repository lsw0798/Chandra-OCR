# nvidia rtx 4090 1ea
# chandra-ocr==0.1.8
# torch==2.8.0, torchvision==0.23.0, cu128

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_html
import torch
import pandas as pd

model = AutoModelForImageTextToText.from_pretrained(
    "datalab-to/chandra",
    dtype=torch.bfloat16, #bfloat16 경량화
    device_map="auto",
    )
model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")

pil_image = Image.open("./imgs/image.jpg")
batch = [BatchInputItem(image=pil_image, prompt_type="ocr_layout")]

# 3. 추론
model.eval()
with torch.no_grad():
    result = generate_hf(batch, model)[0]

# 4. 결과 파싱
html_output = parse_html(result.raw) # 결과의 원본(result.raw)을 HTML 형식으로 변환

with open("html_output.html", "w", encoding="utf-8") as f:
    f.write(html_output)

# 5. HTML 테이블을 CSV로 저장
tables = pd.read_html(html_output) # html 형식 테이블을 DataFrame으로 변환
df = tables[0]
df.to_csv('./csv_output.csv', index=False)

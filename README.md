# Text2Image + Image2Text Demo

Repo ini berisi demo sederhana yang menggabungkan:
- Teks → Gambar (Stable Diffusion via diffusers)
- Gambar → Teks (BLIP captioning via transformers)

## Cara jalankan lokal
1. Buat virtualenv & install:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Export token (jika model private atau perlu HF access):



export HF_TOKEN=hf_xxx

3. Jalankan server:



python app.py

Buka http://localhost:5000.

Deploy ke Render

1. Push repo ke GitHub.


2. Buat Web Service baru di Render dan hubungkan repo.


3. Atur Environment Variable HF_TOKEN di dashboard Render.


4. Build command: pip install -r requirements.txt.


5. Start command: python app.py.

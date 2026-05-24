import os
import sys
import torch
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Коректне налаштування шляхів імпорту
# Тепер файл у src/servers/, тому шлях до external/LuxTTS інший
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
external_path = os.path.join(project_root, "external", "LuxTTS")

if external_path not in sys.path:
    sys.path.append(external_path)

try:
    from zipvoice.luxvoice import LuxTTS
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

app = FastAPI(title="LuxTTS Clone Server")

print("--- Ініціалізація LuxTTS (це може зайняти хвилину) ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LuxTTS('YatharthS/LuxTTS', device=device)
print(f"--- LuxTTS завантажено на {device} ---")

class CloneRequest(BaseModel):
    text: str
    ref_path: str
    output_path: str

@app.post("/generate")
async def generate(req: CloneRequest):
    if not os.path.exists(req.ref_path):
        raise HTTPException(status_code=404, detail=f"Reference audio not found: {req.ref_path}")
        
    if not req.text or len(req.text.strip()) < 2:
        raise HTTPException(status_code=400, detail="Text is too short or empty after stripping.")
        
    try:
        prompt = model.encode_prompt(req.ref_path, rms=0.01)
        wav = model.generate_speech(req.text, prompt, num_steps=4)
        wav_np = wav.numpy().squeeze()
        sf.write(req.output_path, wav_np, 48000)
        return {"status": "success", "message": f"Generated: {req.output_path}"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during generation:\n{error_details}")
        # Якщо текст містить символи, які модель не може обробити, це часто викликає ValueError або RuntimeError
        if "No English or Chinese characters found" in str(e) or "empty" in str(e).lower() or "upper bound and lower bound inconsistent" in str(e):
            raise HTTPException(status_code=400, detail=f"Unsupported text content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ready", "device": device}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

import os
import sys
import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MOSS-TTS API Wrapper")

# Оскільки MOSS-TTS має конфліктні залежності, ми обгортаємо його виклик
# але тримаємо сервер запущеним для уніфікації інтерфейсу

class GenerateRequest(BaseModel):
    text: str
    output_path: str
    ref_path: str = None

@app.post("/generate")
async def generate(req: GenerateRequest):
    # Визначаємо шлях до скрипта інференсу
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    moss_dir = os.path.join(project_root, "external", "MOSS-TTS-Nano")
    infer_script = os.path.join(moss_dir, "infer.py")
    
    # Використовуємо поточний python (який у venv)
    venv_python = sys.executable
    
    cmd = [
        venv_python, infer_script,
        "--text", req.text,
        "--output-audio-path", req.output_path,
        "--device", "cpu",
        "--disable-wetext-processing"
    ]
    
    if req.ref_path:
        cmd.extend(["--prompt-audio-path", req.ref_path])
        
    try:
        # Виконуємо інференс
        env = os.environ.copy()
        # Додаємо шлях до MOSS у PYTHONPATH, щоб працювали внутрішні імпорти
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{moss_dir}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = moss_dir
        
        # Запускаємо з таймаутом, щоб не блокувати сервер назавжди
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"MOSS Inference Error Output: {result.stderr}")
            raise HTTPException(status_code=500, detail=result.stderr)
            
        return {"status": "success", "output": req.output_path}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference timeout exceeded")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

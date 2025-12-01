#!/usr/bin/env python3
"""
Test-Skript für Qwen2.5-VL mit vLLM
Testet die grundlegende Funktionalität mit einem einzelnen Bild.
"""

import os
from pathlib import Path
from PIL import Image

# vLLM nur importieren wenn verfügbar (Linux mit CUDA)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ vLLM nicht verfügbar (nur Linux mit CUDA)")

from dotenv import load_dotenv
load_dotenv()

# HuggingFace Login
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ HuggingFace Login erfolgreich")

# Konfiguration
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Projekt-Root ermitteln
_script_path = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("VLM_PROJECT_ROOT", _script_path.parent.parent.parent))
DATA_DIR = PROJECT_ROOT / "data"

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Data Dir: {DATA_DIR}")


def find_test_image():
    """Findet ein Test-Bild aus dem Dataset."""
    # Suche nach PNG/JPG in dataset_final
    dataset_final = DATA_DIR / "dataset_final"
    if dataset_final.exists():
        for img_file in dataset_final.glob("**/*.png"):
            return img_file
        for img_file in dataset_final.glob("**/*.jpg"):
            return img_file
    
    # Fallback: Suche in anderen Ordnern
    for folder in ["Klasse3-4", "Klasse5-6", "Klasse7-8", "Klasse9-10", "Klasse11-13"]:
        folder_path = DATA_DIR / folder
        if folder_path.exists():
            for img_file in folder_path.glob("**/*.png"):
                return img_file
    
    return None


def main():
    print("=" * 60)
    print("🚀 vLLM Test für Qwen2.5-VL-7B")
    print("=" * 60)
    
    if not VLLM_AVAILABLE:
        print("❌ vLLM nicht verfügbar. Bitte auf Linux mit CUDA ausführen.")
        return
    
    # Test-Bild finden
    test_image = find_test_image()
    if not test_image:
        print("❌ Kein Test-Bild gefunden!")
        return
    
    print(f"📷 Test-Bild: {test_image}")
    
    # Bild laden
    image = Image.open(test_image).convert("RGB")
    print(f"   Größe: {image.size}")
    
    # vLLM initialisieren
    print(f"\n🏗️ Lade Modell: {MODEL_ID}")
    print("   (Dies kann einige Minuten dauern...)")
    
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )
    print("✅ Modell geladen!")
    
    # Sampling Parameter
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0,
        top_p=1.0,
    )
    
    # Prompt erstellen
    prompt = """<|im_start|>system
Du bist ein präzises mathematisches Assistenzsystem. 
Analysiere die Aufgabe im Bild und gib die korrekte Antwort.
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im Format: {"answer": "X"} wobei X einer der Buchstaben A, B, C, D oder E ist.
<|im_end|>
<|im_start|>user
<image>
Löse die Mathematik-Aufgabe im Bild. Gib nur das JSON zurück.
<|im_end|>
<|im_start|>assistant
"""
    
    # Inference
    print("\n🧠 Generiere Antwort...")
    
    # vLLM Multi-Modal Input
    outputs = llm.generate(
        [{
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }],
        sampling_params=sampling_params,
    )
    
    # Ergebnis
    generated_text = outputs[0].outputs[0].text
    
    print("\n" + "=" * 60)
    print("📤 ERGEBNIS:")
    print("=" * 60)
    print(f"Raw Output: {generated_text}")
    print("=" * 60)
    
    # Cleanup
    del llm
    print("\n✅ Test abgeschlossen!")


if __name__ == "__main__":
    main()

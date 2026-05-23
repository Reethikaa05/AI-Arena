"""
Modal Serverless Deployment — Qwen2.5-7B-Instruct
Deploys OSS model with auto-scaling GPU, pay-per-use.
Run: modal deploy modal_deploy.py
"""

import modal
import os

app = modal.App("ai-arena-oss")

# GPU image with model pre-cached
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "torch", "accelerate", "fastapi", "uvicorn")
    .run_commands("pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121")
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Cache model at build time
with image.imports():
    from vllm import LLM, SamplingParams

@app.cls(
    gpu=modal.gpu.A10G(),
    image=image,
    container_idle_timeout=300,  # Scale to zero after 5 min idle
    allow_concurrent_inputs=10,
)
class QwenModel:
    @modal.enter()
    def load_model(self):
        self.llm = LLM(model=MODEL_ID, dtype="float16", max_model_len=4096)
        self.params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

    @modal.method()
    def generate(self, messages: list) -> dict:
        import time
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        t0 = time.time()
        outputs = self.llm.generate([prompt], self.params)
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        return {
            "response": outputs[0].outputs[0].text,
            "latency_ms": latency_ms,
            "model": MODEL_ID,
            "provider": "Modal (A10G GPU)",
        }


# FastAPI wrapper
from fastapi import FastAPI
from pydantic import BaseModel

web_app = FastAPI()

class ChatReq(BaseModel):
    messages: list
    
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    model = QwenModel()
    
    @web_app.post("/chat")
    async def chat(req: ChatReq):
        return model.generate.remote(req.messages)
    
    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}
    
    return web_app


# Cost estimate: A10G = ~$1.10/hr on Modal
# At 1s avg latency per request: ~3600 requests/hr
# Cost per 1000 requests: ~$0.31
# Idle: $0 (scales to zero)

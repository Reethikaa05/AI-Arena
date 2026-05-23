"""
HuggingFace Spaces Deployment — Qwen2.5-0.5B-Instruct (free tier)
Deploy this file to a HuggingFace Space as app.py with SDK: gradio
"""

import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
print("Model loaded!")

SYSTEM = "You are a helpful, honest, and harmless AI assistant. You refuse harmful requests politely."

def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

demo = gr.ChatInterface(
    fn=chat,
    title="⚡ AI Arena — OSS Assistant (Qwen2.5)",
    description="Powered by Qwen2.5-0.5B-Instruct · Open Source · HuggingFace Spaces",
    examples=[
        "What is the capital of Australia?",
        "Explain quantum computing in simple terms",
        "Write a haiku about artificial intelligence",
    ],
    theme=gr.themes.Soft(primary_hue="emerald"),
)

if __name__ == "__main__":
    demo.launch()

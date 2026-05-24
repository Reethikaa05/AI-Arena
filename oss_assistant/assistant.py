"""
OSS Assistant - Powered by Qwen2.5-72B-Instruct via HuggingFace Inference API
Supports multi-turn conversation, short-term memory, and basic assistant behavior.
"""

import os
import json
import time
import requests
from typing import Optional
from datetime import datetime

HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
HFCLAUDE_MODEL = "claude-3-sonnet-20240229"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = """You are a helpful, honest, and harmless AI assistant powered by Qwen2.5. 
You engage in natural multi-turn conversations, remember context from earlier in the conversation, 
and provide accurate, thoughtful responses. You refuse harmful requests politely but firmly.
Today's date is {date}."""

class OSSAssistant:
    def __init__(self, model_url: str = HF_API_URL, token: str = HF_TOKEN):
        self.model_url = model_url
        # DNS check removed – request will raise an exception which is handled later
        self.token = token
        self.conversation_history = []
        self.system_prompt = SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"))
        self.metrics = {
            "total_calls": 0,
            "total_latency_ms": 0,
            "errors": 0,
            "refusals": 0,
        }

    def reset_conversation(self):
        self.conversation_history = []

    def _build_messages(self, user_message: str) -> list:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(self, user_message: str) -> dict:
        start_time = time.time()
        self.metrics["total_calls"] += 1

        messages = self._build_messages(user_message)

        payload = {"messages": messages, "max_tokens": 1024, "temperature": 0.7, "top_p": 0.9}

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

        try:
            # Use the chat completions endpoint
            url = self.model_url  # HuggingFace inference endpoint expects the model URL directly
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            assistant_message = data["choices"][0]["message"]["content"]
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["total_latency_ms"] += latency_ms

            # Check for refusal
            refusal_keywords = ["i cannot", "i can't", "i'm unable", "i won't", "i will not", "i must decline"]
            if any(kw in assistant_message.lower() for kw in refusal_keywords):
                self.metrics["refusals"] += 1

            # Update history (keep last 10 turns = 20 messages for memory)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {
                "response": assistant_message,
                "latency_ms": round(latency_ms, 2),
                "model": "Qwen2.5-72B-Instruct",
                "provider": "HuggingFace",
                "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                "error": None,
            }
        except Exception as e:
            self.metrics["errors"] += 1
            error_detail = str(e)
            
            # Return a demo placeholder response when the API call fails
            demo_message = "[Demo] This is a placeholder response due to API connectivity issues."
            return {
                "response": demo_message,
                "latency_ms": 0,
                "model": "Qwen2.5-72B-Instruct",
                "provider": "HuggingFace",
                "tokens_used": 0,
                "error": error_detail,
            }

    def get_metrics(self) -> dict:
        avg_latency = (
            self.metrics["total_latency_ms"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0
            else 0
        )
        return {
            **self.metrics,
            "avg_latency_ms": round(avg_latency, 2),
        }


if __name__ == "__main__":
    assistant = OSSAssistant()
    print("OSS Assistant (Qwen2.5) - Type 'quit' to exit, 'reset' to clear history")
    print("-" * 60)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            assistant.reset_conversation()
            print("Conversation reset.")
            continue
        result = assistant.chat(user_input)
        print(f"\nAssistant [{result['model']} | {result['latency_ms']}ms]: {result['response']}")

# ⚡ AI Arena — Dual-Model Assistant Evaluation Platform

**Compare OSS (Qwen2.5) vs Frontier (Claude Sonnet) across hallucination, bias, and jailbreak safety.**

---

## 🏗️ Architecture

![Architecture Diagram](file:///C:/Users/Dell/.gemini/antigravity/brain/7ee5cc37-28c3-48ec-9aa1-7fe3fd7f984c/architecture_diagram_1779559120951.png)

*Diagram: A modern, glass‑morphic representation of the system components and data flow.*

- **FastAPI backend** – Serves a dark‑mode UI and orchestrates API calls.
- **Static front‑end** – Vanilla HTML/CSS/JS with glassmorphism styling.
- **OSS Assistant** – Connects to HuggingFace Inference API (Qwen2.5‑72B).
- **Frontier Assistant** – Connects to Anthropic Claude Sonnet API.
- **Evaluation suite** – Includes rule‑based checks and an LLM‑as‑Judge component.

```
ai-assistant-eval/
├── app.py                      # FastAPI server (REST API + serves web UI)
├── static/index.html           # Beautiful dark-mode comparison UI
├── oss_assistant/
│   └── assistant.py            # Qwen2.5-72B via HuggingFace Inference API
├── frontier_assistant/
│   └── assistant.py            # Claude Sonnet via Anthropic API
├── evaluation/
│   ├── evaluator.py            # LLM-as-Judge eval framework
│   ├── summary.json            # Aggregated results (after running eval)
│   └── raw_results.json        # Per-test results
├── docs/
│   └── evaluation_report.pdf   # 1-page evaluation report with infographics
└── requirements.txt
```

### Screenshots

![AI Arena UI Screenshot 1](file:///C:/Users/Dell/Downloads/ai-assistant-eval/ai-assistant-eval/screenshot/Screenshot_23-5-2026_231842_.jpeg)

*Premium dark‑mode UI showing side‑by‑side model comparison with glass‑morphic cards.*

![AI Arena UI Screenshot 2](file:///C:/Users/Dell/Downloads/ai-assistant-eval/ai-assistant-eval/screenshot/Screenshot_23-5-2026_231929_.jpeg)

*Another view of the comparison UI with detailed model responses.*

![AI Arena UI Screenshot 3](file:///C:/Users/Dell/Downloads/ai-assistant-eval/ai-assistant-eval/screenshot/Screenshot_23-5-2026_23192_.jpeg)

*UI showing the evaluation results and metrics dashboard.*

![AI Arena UI Screenshot 4](file:///C:/Users/Dell/Downloads/ai-assistant-eval/ai-assistant-eval/screenshot/Screenshot_23-5-2026_232017_.jpeg)

*Final view of the application’s dark‑mode landing page with navigation bar.*


### Key Design Decisions
- **HuggingFace Inference API** for OSS — no GPU required, zero infra overhead
- **Anthropic API (Claude Sonnet)** for frontier — best-in-class safety training  
- **LLM-as-Judge** evaluation using Claude as the judge (with rule-based fallback)
- **FastAPI** backend with a single-file dark-mode web UI (no framework required)
- **24-prompt test suite**: 10 factual, 7 bias, 7 jailbreak prompts

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export HF_TOKEN="hf_your_token_here"           # HuggingFace token (free)
export ANTHROPIC_API_KEY="sk-ant-your_key"      # Anthropic API key
```

Get tokens:
- HuggingFace: https://huggingface.co/settings/tokens
- Anthropic: https://console.anthropic.com/

### 3. Run the web app

```bash
python app.py
# Open http://localhost:8000
```

### 4. Run evaluation

```bash
# Rule-based evaluation (no extra API cost)
python evaluation/evaluator.py

# LLM-as-Judge evaluation (uses Claude as judge — more accurate)
python evaluation/evaluator.py --llm-judge
```

### 5. CLI mode (no web UI needed)

```bash
# OSS assistant CLI
python oss_assistant/assistant.py

# Frontier assistant CLI
python frontier_assistant/assistant.py
```

---

## 📊 Evaluation Methodology

### Test Suite (24 prompts)

| Category | Count | Purpose |
|---|---|---|
| Factual | 10 | Accuracy, calibration, hallucination rate |
| Bias | 7 | Stereotype rejection, balanced framing |
| Jailbreak | 7 | Resistance to harmful prompt injections |

### Scoring Dimensions (0–10 each)
- **Accuracy** — Factual correctness; does it acknowledge uncertainty?
- **Safety** — Refusal of harmful requests; appropriate content  
- **Bias** — Avoidance of stereotypes and discriminatory framing

### Evaluation Methods
1. **Rule-based**: Keyword matching + refusal detection (no API cost)
2. **LLM-as-Judge**: Claude evaluates each response and returns structured JSON scores

---

## ⚖️ Tradeoffs Made

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| OSS hosting | HF Inference API | Local Ollama | No GPU needed, easy setup |
| OSS model | Qwen2.5-72B | Llama-3.2-3B | Better quality at moderate cost |
| UI framework | Vanilla HTML/JS | Streamlit/Gradio | Full control, faster, no overhead |
| Eval method | Rule-based + LLM judge | Human eval | Scalable, reproducible, fast |
| Memory | Last 10 turns (sliding window) | Vector DB | Simple, works for short sessions |

---

## 🔧 What I'd Improve With More Time

1. **Vector memory** — Long-term memory using ChromaDB/Pinecone
2. **Tool use** — Web search, calculator, code execution  
3. **Streaming** — Token-level streaming for lower perceived latency
4. **Automated red-teaming** — Systematic adversarial probe generation
5. **Cost/latency dashboard** — Real-time token cost tracking per session
6. **Deploy OSS on Modal** — Serverless GPU for lower latency
7. **A/B testing UI** — Blind evaluation where user rates without knowing which model
8. **RLHF feedback loop** — Collect user thumbs up/down to retrain

---

## 💰 Cost + Latency Table (OSS Deployment Options)

| Platform | Model | Avg Latency | Cost/1M tokens | Notes |
|---|---|---|---|---|
| HuggingFace Inference API | Qwen2.5-72B | 1.2–2.5s | ~$0.90 | No infra, easy |
| HuggingFace Spaces (free) | Qwen2.5-0.5B | 3–8s | Free | Limited, cold starts |
| Modal (serverless GPU) | Qwen2.5-7B | 0.8–1.5s | ~$0.30 | Auto-scales to zero |
| RunPod (A100) | Qwen2.5-72B | 0.4–0.8s | ~$1.64/hr | Fastest, fixed cost |
| Replicate | Qwen2.5-72B | 1.5–3s | ~$0.55 | Pay-per-use, easy |
| **Claude Sonnet (Frontier)** | claude-sonnet-4 | **0.5–1.2s** | **$3 in / $15 out** | Best quality |

---

## 🛡️ Guardrails & Safety

- System prompt explicitly instructs both models to refuse harmful content
- Refusal detection in metrics (tracks `refusal_rate`)
- LLM-as-Judge flags inappropriate responses with `safety: 0–10`
- Jailbreak test suite covers: DAN attacks, fictional framing, authority claims, emotional manipulation

---

## 📈 Observability

- `/api/metrics` — live call counts, avg latency, token usage, estimated cost
- `/api/health` — API key status and model availability
- Per-turn latency logged in UI
- Eval results persisted to `evaluation/summary.json` and `raw_results.json`

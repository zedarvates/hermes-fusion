# Hermes Fusion

**Multi-LLM Fusion Engine for Hermes Agent** — Local cluster + Cloud orchestration

Hermes Fusion orchestrates multiple LLM providers (LocalAI, xAI Grok, OpenAI, Anthropic, Hailo-8 vision) through pluggable fusion strategies to produce higher-quality, more reliable answers than any single model.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Fusion Engine                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Providers  │  │  Strategies │  │    Semantic Cache   │  │
│  ├─────────────┤  ├─────────────┤  ├─────────────────────┤  │
│  │ LocalAI     │  │ Weighted    │  │ Qdrant (HNSW)       │  │
│  │ xAI Grok    │  │ Vote        │  │ TTL + similarity    │  │
│  │ OpenAI GPT4o│  │ Best-of-N   │  │ threshold 0.92      │  │
│  │ Anthropic   │  │ CoT Consensus│  │                     │  │
│  │ Hailo-8     │  │ Handoff     │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Local-first**: Primary inference on EUREKAI (LocalAI, Hailo-8, Qdrant)
- **Cloud fallback**: xAI Grok, OpenAI GPT-4o, Anthropic Claude for complex reasoning
- **4 Fusion Strategies**:
  - `weighted_vote` — Provider reliability weighted consensus
  - `handoff` — Sequential fallback (local → cloud)
  - `cot_consensus` — Chain-of-thought majority agreement
  - `best_of_n` — Judge LLM picks best answer
- **Semantic Cache**: Qdrant HNSW vector search, 92% similarity threshold, 24h TTL
- **Vision**: Hailo-8 classify/detect/OCR via MCP
- **Zero secrets in repo**: All keys via env vars (`.env`)
- **CLI + Library**: `hermes-fusion query "..."` or `from hermes_fusion import FusionEngine`

## Quick Start

```bash
# Install
pip install hermes-fusion[cloud,qdrant] --break-system-packages

# Configure
cp config.example.toml config.toml
cp .env.example .env
# Edit .env with your API keys

# Run
hermes-fusion query "What is quantum entanglement?"
hermes-fusion health
hermes-fusion strategies
```

## Usage

### CLI

```bash
# Query with default strategy (weighted_vote)
hermes-fusion query "Explain Rust ownership"

# Use specific strategy
hermes-fusion query "Complex math problem" --strategy cot_consensus

# JSON output for scripting
hermes-fusion query "API design question" --json

# Health check
hermes-fusion health

# Cache management
hermes-fusion cache cleanup --hours 48

# List strategies
hermes-fusion strategies
```

### Python Library

```python
import asyncio
from hermes_fusion import FusionEngine, FusionConfig

async def main():
    # Load from config.toml
    config = FusionConfig.from_toml("config.toml")
    
    # Create engine (providers auto-initialized from config)
    engine = await create_engine(config)
    
    # Query
    result = await engine.query(
        "How to optimize Redis memory?",
        strategy="weighted_vote"
    )
    
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Providers: {result.participating_providers}")

asyncio.run(main())
```

## Configuration

All settings in `config.toml`:

```toml
[fusion]
default_strategy = "weighted_vote"
max_parallel_providers = 3
timeout_seconds = 120

[providers.localai]
base_url = "http://192.168.1.47:8080"

[providers.cloud.xai]
api_key_env = "XAI_API_KEY"
model = "grok-3"
weight = 1.5
```

Environment variables in `.env`:

```bash
XAI_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

## Providers

| Provider | Type | Models | Use Case |
|----------|------|--------|----------|
| LocalAI | Local | Gemma-4, GPT-4o, Whisper, TTS | Primary chat, STT, TTS |
| Hailo-8 | Local | ResNet18, YOLOv8, OCR | Vision inference |
| xAI Grok | Cloud | Grok-3 | Complex reasoning |
| OpenAI | Cloud | GPT-4o, embeddings | General purpose, embeddings |
| Anthropic | Cloud | Claude-3.5-Sonnet | Long context, analysis |

## Fusion Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `weighted_vote` | Weighted consensus by provider reliability | General purpose |
| `handoff` | Sequential: local → cloud fallback | Cost optimization |
| `cot_consensus` | Extract answers from CoT, majority vote | Reasoning tasks |
| `best_of_n` | Judge LLM evaluates all answers | Quality-critical |

## Infrastructure (EUREKAI Cluster)

```
EUREKAI (192.168.1.47, Ubuntu + GPU)
├── LocalAI:8080        (Gemma-4, Whisper, TTS)
├── ComfyUI:8188        (Stable Diffusion)
├── Qdrant:6333         (Vector DB, 768-dim)
├── Hailo-8:8767        (Vision accelerator)
└── Postgres:5432       (Business data)
```

## Requirements

- Python 3.11+
- EUREKAI services running (LocalAI, Qdrant, Hailo-8)
- Optional: Cloud API keys for xAI, OpenAI, Anthropic

## Development

```bash
# Install dev dependencies
pip install -e .[dev] --break-system-packages

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

## License

MIT — See LICENSE for details.

## Author

Sylvain Galliez (zedarvates) — Built for Hermes Agent ecosystem
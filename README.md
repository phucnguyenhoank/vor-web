# VOR-Web

This project is managed with [uv](https://docs.astral.sh/uv/), so install it first.

## Installation

### Install uv (Linux)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

For other platforms and installation options, see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/).

### Clone and set up environment

```bash
git clone https://github.com/phucnguyenhoank/vor-web.git
cd vor-web
uv sync
```

### Download required models

You need the following models inside the `vor-web` folder:

* [`faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3)
* [`HumAwareVad`](https://github.com/CuriousMonkey7/HumAwareVad)

```bash
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-large-v3
git clone https://github.com/CuriousMonkey7/HumAwareVad.git
```

## Run

```bash
uv run enagent.py
```

Then open: [http://localhost:7860](http://localhost:7860)

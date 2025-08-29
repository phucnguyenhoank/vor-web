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


Since I’ve already modified the original **HumAwareVad** code to better fit this project, it is included directly in this repository. You only need to download the additional model below:

```bash
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-large-v3
```

For the final setup step, you’ll also need an LLM. This project uses [Ollama](https://ollama.com/) to manage and run the model. [Download Ollama](https://ollama.com/download/linux), then open your terminal and pull the required model with:

```bash
ollama pull llama3.1:8b-instruct-q8_0
```


## Run

```bash
uv run enagent.py
```

**Note:**

- It may take a few seconds for the application to fully start.

- If you are running a large LLM on limited hardware, the response time may be longer than expected.

Then open: [http://localhost:7860](http://localhost:7860)

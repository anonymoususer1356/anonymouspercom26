# Model preparation

Model binaries are deliberately excluded from this repository. This directory remains empty in Git; use a local model root or paths supplied on the command line.

## Download and optimize

Install the model extras and download available upstream artifacts:

```bash
pip install -e '.[models]'
python scripts/models/download_raw_models.py --output-root /path/to/models/raw
```

The downloader records source repository, size, and SHA-256 digest in `SOURCE_MANIFEST.json`. It currently covers the upstream LS-EEND, Conv-TasNet, CAM++ and Nemotron artifacts retained for model-selection reproducibility. Sandglasset, ReDimNet-B1, and Moonshine Medium preparation depends on their upstream releases and the conversion scripts used by the experiment.

Optimize downloaded Moonshine ONNX graphs for ONNX Runtime:

```bash
python scripts/models/prepare_moonshine_ort.py --model-dir /path/to/moonshine-medium-int8
```

## SLM

The paper student is Qwen3.5 2B. Base, SFT, and DPO models were merged in BF16, exported to f16 GGUF without the auxiliary MTP head, and quantized with llama.cpp Q4_0. Trained adapters and GGUF files are not published in this repository. Use a compatible local model to exercise the runtime or reproduce training before export.

Expected local artifacts are ignored by Git: `*.onnx`, `*.ort`, `*.pt`, `*.bin`, `*.gguf`, and `*.safetensors`.

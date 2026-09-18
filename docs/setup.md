# Download the model and start the server

The server runs on **macOS with Apple Silicon**. The tested machine is an M2 Ultra with **64 GiB of memory**.
The model uses roughly **18–19 GB of MLX allocations** in our measurements, in addition to the operating system and other applications.
We have not established a minimum supported memory size across Macs.

## One-command start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not already have it.
Git must also be available for this GitHub source installation.
Then run:

```sh
uvx --python 3.13 --from git+https://github.com/Saik0s/diffusiongemma-jev-macos jev-local start
```

`uvx` runs a Python command in an isolated tool environment.
It obtains the Python version and package dependencies needed by the command.
`jev-local start` handles the model and starts the HTTP server.

The initial model download is about **17.85 GB**, or **16.6 GiB**.
The downloader requires space for missing files plus **1 GiB** of reserve.
Allow about **20 GB free**, plus room for Python and dependencies.
The download can take several minutes or longer, depending on your connection.

Startup checks the platform, requested port, model structure, and download destination.
New downloads use an immutable model revision and verify every required file's size and SHA-256 checksum.
Progress messages identify the download, verification, and loading stages.

The server is ready after `Application startup complete` appears.
It listens at **http://127.0.0.1:8017**. Stop it with **Ctrl-C**.

## Run from a checkout

A checkout is useful for the examples, tests, and benchmark reports:

```sh
git clone https://github.com/Saik0s/diffusiongemma-jev-macos.git
cd diffusiongemma-jev-macos
uv run jev-local start
```

To install this checkout into uv's tool environment for a shorter command:

```sh
uv tool install --python 3.13 .
jev-local start
```

These are alternative ways to run the same server. Keep only one process running to avoid loading a second copy of the model.
The remainder of this page uses `uv run` from the checkout.

## How model selection works

`start` checks these locations in order:

1. A directory supplied with `--model`, or `JEV_MODEL_PATH` if the flag is absent.
2. The existing LM Studio directory, `~/.cache/lm-studio/models/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit`.
3. The pinned snapshot in the Hugging Face cache.
4. A download into that cache, if no complete model is available.

An explicitly selected invalid directory is an error; it does not silently download a different model.
An incomplete automatic LM Studio candidate falls through to the managed cache.

**Existing local directories receive structural checks only.** That confirms required files are present, not that their contents match the pinned release.
**Managed snapshots receive size and checksum verification**, including on later starts.
Hashing the large files can take time even when no download is needed.

The download source is [mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit/tree/30f3c7c7746bf41cfd1a290155cc3b777ab588b9),
revision `30f3c7c7746bf41cfd1a290155cc3b777ab588b9`.
The model includes four main weight files and an additional OptiQ file needed by the loader.
You do not need to choose or convert these files yourself.

## Use another path or port

```sh
uv run jev-local start --model /path/to/model
uv run jev-local start --cache-dir /Volumes/Models/huggingface --port 8018
```

`--cache-dir` controls the Hugging Face download cache. It does not override a valid explicit or LM Studio model.
Otherwise the Hugging Face default is normally `~/.cache/huggingface/hub`, respecting `HF_HOME` and `HF_HUB_CACHE`.

To load an existing model without the download fallback:

```sh
uv run jev-local serve --model /path/to/model
```

Both `serve` and `start` accept `--host`, `--port`, and `--max-prompt-tokens`.
The defaults are `127.0.0.1`, `8017`, and `8192`.
Use `--help` for the complete command syntax.

## Work offline or retry a download

Once Python dependencies and the model are cached:

```sh
uv run --offline jev-local start --offline
```

The first `--offline` belongs to uv. The second prevents model downloads.
An incomplete cache produces a clear error instead of starting with missing files.
The inference API needs no internet connection; the optional browser `/docs` interface loads assets from a CDN.

If a download is interrupted, rerun the same online command.
Hugging Face's downloader reuses completed files and manages download locks.
An interrupted file may restart from the beginning; do not assume its partial bytes survive restarting the command.
It honors its standard environment settings, including `HTTPS_PROXY` for a proxy and `HF_HUB_OFFLINE` for offline behavior.
See [the upstream download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

## Troubleshooting

### The port is already in use

Use another port, for example `--port 8018`, or stop the server you previously started with Ctrl-C.
Startup checks the requested listener before downloading the model.
Do not stop an unrelated service just to use the default port.

### The model is incomplete or a checksum does not match

For a user-managed directory, check that it is the complete OptiQ DiffusionGemma checkpoint.
For the managed cache, retry with a fresh `--cache-dir` or repair the affected Hugging Face cache.
The server will not accept a managed file whose checksum differs from the pinned manifest.
An interrupted download can be retried by rerunning `start` online; completed files are reused.

### There is not enough disk space

Free the amount reported by the command, or choose a writable volume with `--cache-dir`.
The model stays outside the Git checkout.
No administrator privileges are needed for a writable user directory.

### Downloading fails behind a proxy

Check the connection and your existing `HTTPS_PROXY` configuration.
Model downloads go to Hugging Face; package installation also needs access to Python package sources and GitHub.
Errors deliberately avoid printing transport details that could contain credentials.
Provisioning uses the standard Hugging Face HTTP transfer path and suppresses dependency transport logs while downloading.
Xet acceleration is disabled for that operation because its separate native logs can contain raw request URLs.

### The machine slows down or loading fails

Close memory-heavy applications and ensure only one server is running.
Our **64 GiB** test machine is the verified configuration, not a claim that every smaller Mac will work.
Use the OptiQ checkpoint and pinned runtime; stock `mlx-lm` and `mlx-vlm` do not load it.

### A request is rejected because the prompt is too long

Supply fewer or shorter snippets, or split the work into multiple requests.
The default **8,192-token** limit applies after formatting the state and questions.
Increasing `--max-prompt-tokens` can increase memory use and latency.
Requests are rejected rather than silently shortened.

## Remove the installation

Stop the running server with Ctrl-C.
If you used `uv tool install`, run `uv tool uninstall diffusion-jev` to remove that tool environment.
A checkout installation is confined to the checkout's `.venv`; `uvx` uses uv's cache.

Model files are separate and remain available after removing the Python tool.
Use Hugging Face's cache tools or move the specific model directory to the Trash when you no longer need it.
A reused LM Studio model may still be used by LM Studio, so keep it if needed there.
The startup command does not modify shell profiles, agent configurations, or launch services.

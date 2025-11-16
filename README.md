# Synthwave (HackRPI 2025 Project)

Physically controlled generative music

[![Logo](./logo.png)](./logo.png)

[Devpost](https://devpost.com/software/synthwave-8fgo0c)

## Setup

- Install [`uv`](https://docs.astral.sh/uv/)
- Install dependencies: `uv sync`
- Run the script: `uv run src/hackrpi_2025/main.py`
- It will send MIDI messages to your computer's first MIDI output device based on the position of your hand.

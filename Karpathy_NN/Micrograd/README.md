# Micrograd — Minimal Autograd from Scratch

This folder contains a compact reimplementation of a tiny Autograd engine (inspired by Andrej Karpathy's "micrograd") plus a small demo showing how forward and backward passes work through a tiny neural network.

**Highlights**
- Implements a `Value` scalar with operator overloading to build a dynamic computation graph.
- Implements backward propagation (autograd) via a topological traversal.
- Includes a small `Neuron`/`Layer`/`MLP` implementation and a runnable demo.

**What you'll find here**
- `micrograd-requirements.txt` — suggested packages (minimal; pure Python).
- `micrograd.ipynb` — notebook exploration and experiments.
- `demo.py` — a self-contained script demonstrating the core `Value` class, a tiny MLP, a training loop, and printed gradients.

Quickstart
1. (Optional) Create and activate a virtualenv, then install any requirements from `micrograd-requirements.txt` if you want to run the notebook.

	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install -r micrograd-requirements.txt
	```

2. Run the demo script to see a minimal autograd and tiny network in action:

	```powershell
	python demo.py
	```

What the demo shows
- Construction of a computational graph by using plain Python operators on `Value` objects.
- Computing a scalar loss, calling `backward()` to populate `.grad` on every `Value`.
- A simple training loop that updates parameters with gradient descent and shows how losses decrease.

Design notes
- The implementation in `demo.py` is intentionally compact and explicit so you can read and follow the forward/backward mechanics easily. If you prefer an interactive exploration, open `micrograd.ipynb`.

Credit
- Inspired by Andrej Karpathy's "micrograd" and the "Neural Networks: Zero to Hero" series.


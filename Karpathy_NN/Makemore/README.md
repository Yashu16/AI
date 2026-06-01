# Makemore - Building a character-level name generator from scratch

Makemore is a hands-on exploration of how a language model can learn the structure of names and generate new ones that feel like they belong in the same distribution. The project starts with a simple bigram model, then moves into a small neural network implemented in PyTorch, making it a clean end-to-end walkthrough of the core ideas behind modern sequence modeling.


## What’s inside

- Character vocabulary building from a names dataset in `names.txt`
- Probability estimation with smoothing to avoid zero-probability transitions
- Random name generation with reproducible sampling
- Negative log-likelihood as the training objective
- A one-hot encoded neural network trained with softmax and manual gradient descent
- L2 regularization to improve generalization and smooth the learned distribution

## How to run

Open `makemore.ipynb` in Jupyter or VS Code and run the cells top to bottom.

Dependencies are standard PyTorch notebook tooling. If you need to recreate the environment, install the packages already used in this workspace's Karpathy notebooks or reuse the existing `.venv` associated with the folder.

## Project focus

This notebook currently covers the transition from a statistical bigram baseline to a trainable neural network for next-character prediction. It is a compact but complete example of how language modeling works at the character level, and it is a good foundation for extending into deeper architectures later.

## Representative result

The notebook includes a full forward/backward training step on the character model. One of the key checkpoints is the loss after the initial softmax pass:

```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

xenc = F.one_hot(xs, num_classes=27).float()
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(len(xs)), ys].log().mean()
print(loss)
```

```text
tensor(3.7292, grad_fn=<NegBackward0>)
```

That makes the learning objective visible and shows the model is being trained with a measurable target that improves over time.


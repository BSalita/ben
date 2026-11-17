# PyTorch trainer for the bidding networks (experiment)

`bidding_nn_torch.py` trains the bidding network (`net=bidding`) or the bidding-info
network (`net=info`) from the same binary arrays as the Keras trainers, with the same
architecture, initial values, optimizer settings and learning rate schedule. It writes the
weights of each epoch as `model/<name>_<date>-E<nn>.npz`; `torch_to_keras.py` turns such a
file into the `.keras` model BEN loads (that step needs TensorFlow, not PyTorch).

```bash
python bidding_nn_torch.py <bin_dir> <name> net=bidding lstm=unrolled compile=2
python torch_to_keras.py model/<name>_<date>-E30.npz <name>_<date>-E30.keras
```

It is an experiment to compare speed, not a replacement: there is no resume from a
checkpoint and no early stopping.

## Measured 2026-11-17

RTX 5080 in a WSL2 container, PyTorch 2.11.0+cu128, cuDNN 9.19, training set on the GPU,
batch sizes as trained for BEN (bidding 512 with 3 x 256 LSTM, info 256 with 3 x 128 LSTM).

| Variant | Bidding, ms per step | Info, ms per step |
|---|---|---|
| `nn.LSTM` (cuDNN) | 8.70 | 7.65 |
| `nn.LSTM`, `amp=1` (float16) | 10.54 | 10.12 |
| `nn.LSTM`, `compile=1` | 8.95 | 7.58 |
| `lstm=unrolled` | 29.77 | 31.62 |
| `lstm=unrolled compile=1` | 12.50 | 12.62 |
| `lstm=unrolled compile=2` (CUDA graphs) | 3.38 | 3.95 |
| Keras, unrolled, 512 steps per XLA program (`gpu_epochs.py` on main) | 2.25 | 0.82 |
| Keras, `model.fit` with the cuDNN LSTM (before 2026-11-17) | 34.5 | 37.8 |

An epoch of the bidding network takes 25 s with the best PyTorch variant (7,361 steps) and
17 s with the Keras loop; for the info network 58 s against 12 s. The step is dominated by
the cost of launching work on the GPU, not by the arithmetic: the Keras loop runs hundreds
of steps as one program, while a compiled PyTorch step is still one forward and one
backward launch per step (compiling several steps into one call changed nothing, because
`backward()` and the optimizer step are not part of the compiled graph).

Check of the trainer and the export, 3 epochs of the bidding network
(`lstm=unrolled compile=2`), against the Keras baseline at the same epoch:

| | PyTorch | Keras baseline |
|---|---|---|
| Training accuracy / loss, epoch 1 | 0.9418 / 0.1731 | 0.9423 / 0.1711 |
| Training accuracy / loss, epoch 3 | 0.9637 / 0.1010 | 0.9640 / 0.1000 |
| Held-out boards, same call | 92.76% | 92.85% |
| Held-out boards, whole auction identical | 50.40% | 50.70% |
| Held-out boards, DD cost, IMPs per board | 2.113 | 2.102 |

A full 30-epoch run was not made with this trainer.

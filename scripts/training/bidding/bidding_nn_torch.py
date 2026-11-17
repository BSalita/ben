"""PyTorch trainer for the bidding and bidding-info networks (experimental, for speed comparison).

Trains the same networks as bidding_nn_keras.py and ../bidding_info/binfo_nn_keras.py from the
same binary arrays (x.npy, y.npy, HCP.npy, SHAPE.npy), and writes the weights of every epoch
as an .npz file. torch_to_keras.py turns such a file into the .keras model BEN loads.

Usage:
  python bidding_nn_torch.py bin_dir name [net=bidding|info] [epochs=30] [batch_size=...]
         [lstm_size=...] [n_layers=3] [learning_rate=...] [decay_steps=...] [amp=0] [compile=0]
         [lstm=cudnn|unrolled] [bench=0]

  net=bidding  defaults as run for BEN: lstm_size=256 batch_size=512 learning_rate=0.001,
               learning rate times 0.96 every decay_steps=3000 batches
  net=info     lstm_size=128 batch_size=256 learning_rate=0.0005, no decay
  amp=1        mixed precision (float16 on the GPU)
  compile=1    torch.compile the training step; compile=2 with mode="reduce-overhead" (CUDA graphs)
  lstm         cudnn: nn.LSTM (the cuDNN kernel); unrolled: the eight rounds as explicit steps
  bench=N      time N training steps after a warm-up, then exit without saving

The whole training set is kept on the GPU. That is possible because the input array is
redundant: all features of a bidding round but the last 4 * 40 are the same in every round of
a hand, and those last ones are four one-hot encoded bids (checked when loading).
"""
import os
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_BIDS = 40
ONE_F16 = 0x3C00   # bit pattern of float16 1.0


def arg_value(name, default):
    value = next((arg.split('=', 1)[1] for arg in sys.argv[3:] if arg.startswith(name + '=')), None)
    return default if value is None else type(default)(value)


def compact_inputs(x, chunk=50_000, threads=8):
    """(examples, rounds, features) float16 -> static (examples, features - 160) float16 and
    bids (examples, rounds, 4) uint8 (index of each one-hot bid, 40 for an all-zero block)."""
    n, n_rounds, n_ftrs = x.shape
    n_static = n_ftrs - 4 * N_BIDS
    static = np.empty((n, n_static), dtype=np.float16)
    bids = np.empty((n, n_rounds, 4), dtype=np.uint8)

    def do(start):
        c = np.asarray(x[start:start + chunk])
        m = len(c)
        s = c[:, 0, :n_static]
        if not (c[:, :, :n_static].view(np.uint16) == s.view(np.uint16)[:, None, :]).all():
            return False
        blocks = np.ascontiguousarray(c[:, :, n_static:]).view(np.uint16).reshape(m, n_rounds, 4, N_BIDS)
        count = np.count_nonzero(blocks, axis=-1)
        if count.max() > 1 or not (blocks.max(axis=-1) == np.where(count == 1, ONE_F16, 0)).all():
            return False
        static[start:start + m] = s
        bids[start:start + m] = np.where(count == 1, blocks.argmax(axis=-1), N_BIDS)
        return True

    with ThreadPoolExecutor(threads) as pool:
        if not all(pool.map(do, range(0, n, chunk))):
            raise SystemExit("x.npy is not of the expected form (static features + four one-hot bids per round)")
    return static, bids


class UnrolledLSTM(nn.Module):
    """One LSTM layer written as explicit steps, with the parameters of nn.LSTM."""
    def __init__(self, n_in, hidden):
        super().__init__()
        self.hidden = hidden
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * hidden, n_in))
        self.weight_hh_l0 = nn.Parameter(torch.empty(4 * hidden, hidden))
        self.bias_ih_l0 = nn.Parameter(torch.zeros(4 * hidden))
        self.bias_hh_l0 = nn.Parameter(torch.zeros(4 * hidden))

    def forward(self, x):
        n, n_rounds, _ = x.shape
        h = x.new_zeros(n, self.hidden)
        c = x.new_zeros(n, self.hidden)
        xw = F.linear(x, self.weight_ih_l0, self.bias_ih_l0 + self.bias_hh_l0)
        out = []
        for t in range(n_rounds):
            i, f, g, o = (xw[:, t] + F.linear(h, self.weight_hh_l0)).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            out.append(h)
        return torch.stack(out, dim=1), None


class Net(nn.Module):
    def __init__(self, n_ftrs, hidden, n_layers, head_sizes, lstm='cudnn'):
        super().__init__()
        make = (lambda n_in: nn.LSTM(n_in, hidden, batch_first=True)) if lstm == 'cudnn' else (lambda n_in: UnrolledLSTM(n_in, hidden))
        self.lstms = nn.ModuleList([make(n_ftrs if i == 0 else hidden) for i in range(n_layers)])
        # Keras BatchNormalization defaults: momentum 0.99 (0.01 here), epsilon 0.001
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden, eps=1e-3, momentum=0.01) for _ in range(n_layers)])
        self.drop = nn.Dropout(0.2)
        self.heads = nn.ModuleList([nn.Linear(hidden, size) for size in head_sizes])
        # Keras initial values: Glorot uniform kernels, orthogonal recurrent kernels, zero biases
        # with the forget gate at 1
        for layer in self.lstms:
            nn.init.xavier_uniform_(layer.weight_ih_l0)
            for k in range(4):
                nn.init.orthogonal_(layer.weight_hh_l0.data[k * hidden:(k + 1) * hidden])
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
            layer.bias_ih_l0.data[hidden:2 * hidden] = 1
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        for lstm, norm in zip(self.lstms, self.norms):
            x, _ = lstm(x)
            x = self.drop(x)
            n, n_rounds, hidden = x.shape
            x = norm(x.reshape(n * n_rounds, hidden)).reshape(n, n_rounds, hidden)
        return [head(x) for head in self.heads]


def keras_weights(model):
    """Weights in the order and layout of the Keras model's get_weights()."""
    w = {}
    k = 0
    def add(t):
        nonlocal k
        w[f'w{k:02d}'] = t.detach().float().cpu().numpy()
        k += 1
    for lstm, norm in zip(model.lstms, model.norms):
        add(lstm.weight_ih_l0.t())
        add(lstm.weight_hh_l0.t())
        add(lstm.bias_ih_l0 + lstm.bias_hh_l0)
        add(norm.weight); add(norm.bias); add(norm.running_mean); add(norm.running_var)
    for head in model.heads:
        add(head.weight.t())
        add(head.bias)
    return w


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    bin_dir, name = sys.argv[1], sys.argv[2]
    net = arg_value('net', 'bidding')
    bidding = net == 'bidding'
    epochs = arg_value('epochs', 30)
    batch_size = arg_value('batch_size', 512 if bidding else 256)
    lstm_size = arg_value('lstm_size', 256 if bidding else 128)
    n_layers = arg_value('n_layers', 3)
    learning_rate = arg_value('learning_rate', 0.001 if bidding else 0.0005)
    decay_steps = arg_value('decay_steps', 3000 if bidding else 0)
    amp = arg_value('amp', 0)
    use_compile = arg_value('compile', 0)
    lstm = arg_value('lstm', 'cudnn')
    bench = arg_value('bench', 0)
    dev = torch.device('cuda')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.time()
    x = np.load(os.path.join(bin_dir, 'x.npy'), mmap_mode='r')
    static, bids = compact_inputs(x)
    n_examples, n_rounds, n_ftrs = x.shape
    g_static = torch.from_numpy(static).to(dev)
    g_bids = torch.from_numpy(bids).to(dev)
    if bidding:
        y = np.load(os.path.join(bin_dir, 'y.npy'), mmap_mode='r')
        ycls = np.empty(y.shape[:2], dtype=np.uint8)
        for a in range(0, n_examples, 500_000):
            c = np.asarray(y[a:a + 500_000])
            if not (c.sum(axis=2) == 1).all():
                raise SystemExit("y.npy has rows that are not one-hot")
            ycls[a:a + 500_000] = c.argmax(axis=2)
        targets = [torch.from_numpy(ycls).to(dev)]
        head_sizes = [N_BIDS]
    else:
        targets = [torch.from_numpy(np.load(os.path.join(bin_dir, f))).to(dev) for f in ('HCP.npy', 'SHAPE.npy')]
        head_sizes = [t.shape[2] for t in targets]
    steps_per_epoch = n_examples // batch_size
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}; {n_examples} examples loaded in {time.time() - t0:.0f} s")
    print(f"net={net} lstm_size={lstm_size} n_layers={n_layers} batch_size={batch_size} steps_per_epoch={steps_per_epoch} "
          f"learning_rate={learning_rate} decay_steps={decay_steps} amp={amp} compile={use_compile} lstm={lstm}", flush=True)

    model = Net(n_ftrs, lstm_size, n_layers, head_sizes, lstm).to(dev)
    # Keras Adam uses epsilon 1e-7
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7, fused=True)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_steps, gamma=0.96) if decay_steps else None
    scaler = torch.amp.GradScaler('cuda', enabled=bool(amp))

    def make_batch(idx):
        s = g_static[idx].float()
        b = F.one_hot(g_bids[idx].long(), N_BIDS + 1)[..., :N_BIDS].float()
        return torch.cat([s[:, None, :].expand(-1, n_rounds, -1), b.reshape(len(idx), n_rounds, 4 * N_BIDS)], dim=2)

    def loss_and_metrics(idx):
        with torch.autocast('cuda', dtype=torch.float16, enabled=bool(amp)):
            out = model(make_batch(idx))
        out = [o.float() for o in out]
        if bidding:
            t = targets[0][idx].long()
            loss = F.cross_entropy(out[0].reshape(-1, N_BIDS), t.reshape(-1))
            stats = [loss.detach(), (out[0].argmax(dim=2) == t).float().mean()]
        else:
            hcp, shape = targets[0][idx].float(), targets[1][idx].float()
            l_hcp, l_shape = F.l1_loss(out[0], hcp), F.l1_loss(out[1], shape)
            loss = l_hcp + l_shape
            # what Keras reports as "accuracy" for these outputs: same argmax as the target
            stats = [loss.detach(), l_hcp.detach(), l_shape.detach(),
                     (out[0].argmax(dim=2) == hcp.argmax(dim=2)).float().mean(),
                     (out[1].argmax(dim=2) == shape.argmax(dim=2)).float().mean()]
        return loss, torch.stack(stats)
    stat_names = ['loss', 'accuracy'] if bidding else ['loss', 'hcp_output_loss', 'shape_output_loss', 'hcp_output_accuracy', 'shape_output_accuracy']

    def train_step(idx):
        loss, stats = loss_and_metrics(idx)
        opt.zero_grad(set_to_none=True)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        return stats
    step = train_step
    if use_compile:
        step = torch.compile(train_step, mode="reduce-overhead" if use_compile == 2 else "default")

    if bench:
        model.train()
        order = torch.randint(0, n_examples, (bench + 60, batch_size), device=dev)
        for i in range(60):
            step(order[i])
        torch.cuda.synchronize()
        t = time.perf_counter()
        for i in range(60, 60 + bench):
            step(order[i])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        print(f"BENCH net={net} lstm={lstm} lstm_size={lstm_size} batch_size={batch_size} amp={amp} compile={use_compile}: "
              f"{1000 * dt / bench:.2f} ms/step, {bench * batch_size / dt:.0f} examples/s, epoch {steps_per_epoch * dt / bench:.0f} s")
        return

    os.makedirs('model', exist_ok=True)
    model_name = f'{name}_{datetime.datetime.now().strftime("%Y-%m-%d")}'
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}", flush=True)
        model.train()
        t_epoch = time.time()
        order = torch.randperm(n_examples, device=dev)[:steps_per_epoch * batch_size].view(steps_per_epoch, batch_size)
        total = torch.zeros(len(stat_names), device=dev)
        for i in range(steps_per_epoch):
            total += step(order[i])
            if sched:
                sched.step()
            if (i + 1) % (steps_per_epoch // 10) == 0 or i + 1 == steps_per_epoch:
                elapsed = time.time() - t_epoch
                means = (total / (i + 1)).tolist()
                print(f"{i + 1}/{steps_per_epoch} - {elapsed:.0f}s {1000 * elapsed / (i + 1):.2f}ms/step - "
                      + " - ".join(f"{k}: {v:.4f}" for k, v in sorted(zip(stat_names, means))), flush=True)
        path = os.path.join('model', f'{model_name}-E{epoch:02d}.npz')
        np.savez(path, net=net, lstm_size=lstm_size, n_layers=n_layers, n_ftrs=n_ftrs, **keras_weights(model))
        print(f"Saved {path}")
        print(f"Epoch took {time.time() - t_epoch:0.4f} seconds", flush=True)


if __name__ == '__main__':
    main()

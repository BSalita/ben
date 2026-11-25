#!/usr/bin/env python3
"""Fine-tune the network the bidding search uses for partner and opponents on Brill's robots' own answers.

The search's continuations ask the Brill-21GF network for North's, East's and West's later calls
(BotBid.bidding_rollout). net_robot_agreement.py showed that it makes only about half of the
robots' bids and doubles in contested later rounds (missed-game leak, 2026-11-23). Here every
answer in the shared cache of Brill's /bid service becomes one training sequence, built with the
inference input (binary.get_auction_binary_sampling, 8 steps padded as BEN's training data is),
with the robot's answer as the only target (the earlier calls of that seat are sometimes another
player's, so they carry no target).

Answers asked in the auctions of the missed-game positions (--tags, --nodes) are split 'leak' and
hands that appear in the past Brill boards (field.jsonl) split 'past', both kept out of training,
so a replay A/B on those boards is not trained on; of the rest, one hand in ten is held out
('test'), all of a hand's turns together.

  python continuation_net.py build --cache robot_calls.jsonl --field field.jsonl --tags tags.jsonl --nodes nodes.jsonl --out DATA_DIR
  python continuation_net.py train --data DATA_DIR --base BASE.keras --out RUN_DIR [--epochs 4] [--lr 3e-4] [--later-weight 1]

train writes RUN_DIR/epoch-<k>.keras and RUN_DIR/eval.json (agreement at the target step, legal
calls only, by group, for the base network as epoch 0 and after every epoch).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
SEATS = 'NESW'
VULN = {'None': [False, False], 'NS': [True, False], 'EW': [False, True], 'All': [True, True]}
FROM_CTX = {'--': 'PASS', 'Db': 'X', 'Rd': 'XX'}
N_STEPS = 8


def norm(hand):
    return '.'.join(''.join(sorted(s, key='AKQJT98765432'.index)) for s in hand.split('.'))


def src_path():
    for d in (os.path.join(HERE, '..', '..', '..', 'src'), '/ben/src'):
        if os.path.exists(os.path.join(d, 'binary.py')):
            return os.path.abspath(d)
    raise SystemExit('BEN src not found')


def build(args):
    import numpy as np
    sys.path.insert(0, src_path())
    from bidding import bidding
    import binary

    models = SimpleNamespace(model_version=3, ns=1, ew=1)      # what the rollouts pass (012.conf: NS = EW = 1)
    parse_hand = binary.parse_hand_f(24)
    past = set()
    for line in open(args.field, encoding='utf-8'):
        b = json.loads(line)
        past.update(norm(h) for h in b['deal'][2:].split())
    # answers asked in the auctions of the missed-game positions (sample_check.py): split 'leak', never trained on
    leak_keys, leak_prefixes = set(), set()
    if args.tags:
        leak_keys = {json.loads(line)['key'] for line in open(args.tags)}
    if args.nodes:
        to_ctx = {'P': '--', 'X': 'Db', 'XX': 'Rd'}
        for line in open(args.nodes, encoding='utf-8'):
            nd = json.loads(line)
            if not nd.get('meta') and len(nd['prefix']) >= 4:     # a short prefix would take in most auctions
                leak_prefixes.add((nd['dealer'], nd['vul'], ''.join(to_ctx.get(c, c) for c in nd['prefix'])))
    entries = {}
    for line in open(args.cache, encoding='utf-8'):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        entries[r['key']] = r['bid']
    keys = sorted(entries)
    n = len(keys)
    print(f'{n} cached answers, {len(past)} past-board hands', flush=True)
    x = np.zeros((n, N_STEPS, 193), dtype=np.float16)
    y = np.zeros((n, N_STEPS, 40), dtype=np.float16)
    meta = {k: np.zeros(n, dtype=np.int8) for k in ('split', 'seat', 'steps', 'contested', 'kind', 'target')}
    legal = np.zeros((n, 40), dtype=bool)
    legal_cache = {}
    t0 = time.time()
    keep = np.ones(n, dtype=bool)
    for i, key in enumerate(keys):
        seat_s, hand, dealer_s, vul, scoring, ctx = key.split('|')
        seat, dealer = SEATS.index(seat_s), SEATS.index(dealer_s)
        calls = [FROM_CTX.get(ctx[j:j + 2], ctx[j:j + 2]) for j in range(0, len(ctx), 2)]
        auction = ['PAD_START'] * dealer + calls
        k = seat
        while k < len(auction) and auction[k] == 'PAD_START':
            k += 4
        steps = 1 + (len(auction) - k) // 4
        real = entries[key]
        real = 'PASS' if real == 'P' else real
        if steps > N_STEPS or real not in bidding.BID2ID:
            keep[i] = False
            continue
        h = np.array(parse_hand(hand), dtype=np.float32).reshape(1, 24)
        x[i] = binary.get_auction_binary_sampling(N_STEPS, list(auction), seat, h, VULN[vul], models, 24)[0]
        t = bidding.BID2ID[real]
        y[i, steps - 1, t] = 1
        ck = (dealer, tuple(calls))
        if ck not in legal_cache:
            legal_cache[ck] = np.array([c == 2 or (c > 2 and bidding.can_bid(bidding.ID2BID[c], auction)) for c in range(40)])
        legal[i] = legal_cache[ck]
        ns = any(c not in ('PASS', 'X', 'XX') for j, c in enumerate(calls) if (dealer + j) % 2 == 0)
        ew = any(c not in ('PASS', 'X', 'XX') for j, c in enumerate(calls) if (dealer + j) % 2 == 1)
        nh = norm(hand)
        in_leak = key in leak_keys or any(d == dealer_s and v == vul and ctx.startswith(c) for d, v, c in leak_prefixes)
        meta['split'][i] = 3 if in_leak else 2 if nh in past else (1 if int(hashlib.md5(nh.encode()).hexdigest(), 16) % 10 == 0 else 0)
        meta['seat'][i] = seat
        meta['steps'][i] = steps
        meta['contested'][i] = ns and ew
        meta['kind'][i] = 0 if real == 'PASS' else 2 if real in ('X', 'XX') else 1
        meta['target'][i] = t
        if i % 50000 == 0:
            print(f'{i}/{n} {time.time() - t0:.0f} s', flush=True)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'x.npy'), x[keep])
    np.save(os.path.join(args.out, 'y.npy'), y[keep])
    np.save(os.path.join(args.out, 'legal.npy'), legal[keep])
    np.savez(os.path.join(args.out, 'meta.npz'), **{k: v[keep] for k, v in meta.items()})
    s = meta['split'][keep]
    print(f'kept {int(keep.sum())}: train {int((s == 0).sum())}, test {int((s == 1).sum())}, past {int((s == 2).sum())}; '
          f'leak {int((s == 3).sum())}; {time.time() - t0:.0f} s', flush=True)


GROUPS = {
    'all': lambda m: np.ones_like(m['seat'], dtype=bool),
    'robot bids': lambda m: m['kind'] == 1,
    'robot doubles': lambda m: m['kind'] == 2,
    'robot passes': lambda m: m['kind'] == 0,
    'contested round 3+': lambda m: (m['contested'] == 1) & (m['steps'] >= 3),
    'contested round 3+ robot bids': lambda m: (m['contested'] == 1) & (m['steps'] >= 3) & (m['kind'] == 1),
    'contested round 3+ robot doubles': lambda m: (m['contested'] == 1) & (m['steps'] >= 3) & (m['kind'] == 2),
    'North contested round 3+ robot bids': lambda m: (m['seat'] == 0) & (m['contested'] == 1) & (m['steps'] >= 3) & (m['kind'] == 1),
    'uncontested round 1': lambda m: (m['contested'] == 0) & (m['steps'] == 1),
}
np = None


def evaluate(model, x, legal, meta, idx_by_split):
    out = {}
    for split, idx in idx_by_split.items():
        pred = model.predict(x[idx], batch_size=4096, verbose=0)
        steps = meta['steps'][idx].astype(int) - 1
        last = pred[np.arange(len(idx)), steps, :]
        last = np.where(legal[idx], last, -1.0)
        same = np.argmax(last, axis=1) == meta['target'][idx]
        sub = {k: v[idx] for k, v in meta.items()}
        out[split] = {g: [int(f(sub).sum()), float(same[f(sub)].mean()) if f(sub).any() else None] for g, f in GROUPS.items()}
    return out


def train(args):
    global np
    import numpy
    np = numpy
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    sys.path.insert(0, src_path())
    from nn.keras_compat import load_model
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    x = np.load(os.path.join(args.data, 'x.npy'))
    y = np.load(os.path.join(args.data, 'y.npy'))
    legal = np.load(os.path.join(args.data, 'legal.npy'))
    meta = dict(np.load(os.path.join(args.data, 'meta.npz')))
    split = meta['split']
    train_idx = np.where(split == 0)[0]
    idx_by_split = {'test': np.where(split == 1)[0], 'past': np.where(split == 2)[0], 'leak': np.where(split == 3)[0]}
    rng = np.random.default_rng(42)
    if args.eval_limit:
        idx_by_split = {k: np.sort(rng.choice(v, min(len(v), args.eval_limit), replace=False)) for k, v in idx_by_split.items()}
    w = np.ones(len(x), dtype=np.float32)
    later = (meta['contested'] == 1) & (meta['steps'] >= 3) & (meta['kind'] != 0)
    w[later] = args.later_weight
    # per-step weights (Keras 3 wants them for a sequence output): only the target step counts; the loss is
    # averaged over all 8 steps, so the weight there is 8 times the example's weight
    sw = np.zeros((len(x), N_STEPS), dtype=np.float32)
    sw[np.arange(len(x)), meta['steps'].astype(int) - 1] = w * N_STEPS
    print(f'train {len(train_idx)}, test {len(idx_by_split["test"])}, past {len(idx_by_split["past"])}; '
          f'weighted (contested round 3+ bids and doubles) {int(later[train_idx].sum())} at {args.later_weight}', flush=True)

    model = load_model(args.base, compile=False)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss='categorical_crossentropy',
                  weighted_metrics=[])
    os.makedirs(args.out, exist_ok=True)
    results = {'args': vars(args), 'epochs': {}}
    t0 = time.time()
    results['epochs']['0'] = evaluate(model, x, legal, meta, idx_by_split)
    print(f"epoch 0 (base): test {results['epochs']['0']['test']}", flush=True)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train_idx)
        model.fit(x[order], y[order], sample_weight=sw[order], batch_size=args.batch, epochs=1, verbose=2, shuffle=False)
        model.save(os.path.join(args.out, f'epoch-{epoch}.keras'))
        results['epochs'][str(epoch)] = evaluate(model, x, legal, meta, idx_by_split)
        print(f"epoch {epoch} ({time.time() - t0:.0f} s): test {results['epochs'][str(epoch)]['test']}", flush=True)
        with open(os.path.join(args.out, 'eval.json'), 'w') as f:
            json.dump(results, f, indent=1)
    report(results)


def report(results):
    epochs = list(results['epochs'])
    for split in results['epochs']['0']:
        print(f'\n{split}: same call at the target step (legal calls only)')
        print(f"{'group':<40} {'n':>7}" + ''.join(f'{"ep " + e:>8}' for e in epochs))
        for g in GROUPS:
            n = results['epochs']['0'][split][g][0]
            vals = [results['epochs'][e][split][g][1] for e in epochs]
            print(f'{g:<40} {n:>7}' + ''.join(f'{100 * v:>7.1f}%' if v is not None else f'{"":>8}' for v in vals))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('mode', choices=['build', 'train', 'report'])
    p.add_argument('--cache')
    p.add_argument('--field')
    p.add_argument('--tags', help="sample_check.py's tags.jsonl: these answers go to split 'leak'")
    p.add_argument('--nodes', help="leak_diagnosis.py's nodes.jsonl: answers after these prefixes go to split 'leak'")
    p.add_argument('--data')
    p.add_argument('--out')
    p.add_argument('--base')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--later-weight', type=float, default=1.0)
    p.add_argument('--eval-limit', type=int, default=0)
    args = p.parse_args()
    if args.mode == 'build':
        return build(args)
    if args.mode == 'train':
        return train(args)
    report(json.load(open(os.path.join(args.out, 'eval.json'))))


if __name__ == '__main__':
    sys.exit(main())

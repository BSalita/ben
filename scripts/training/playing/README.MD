# Training a Neural Network to play

When playing we are using 5 different neural networks in a combination with a double dummy solver.

We have one neural network for finding the opening lead, and a neural network to find the actual play for each position, but this is only used if the there are equal cards from the Double Summy solver.

The NN for each position is to find:

Agreements like top-of-sequence, high from doubleton, top-of-nothing, etc. are picked up by the neural networks if they are in the data.
Also general patterns like "second hand low", "third hand high", "cover the honor", "draw trumps", "finesse" are picked up by the cardplay NN.



### Instructions

You need to be located in the `scripts/training/playing` directory when you execute the following steps.

First, activate the environment

```
conda activate ben
```

You should have a `playingdata.txt` (not included in this repository) file now which contains deals and their corresponding auctions like this:

```
W:AT73.JT4.92.A984 K4.AK76.KQT85.T2 QJ9865.Q53.64.65 2.982.AJ73.KQJ73
W ALL 3N.-1.N
PP 1N PP 3N PP PP PP
SQS2SAS4S3SKS5H2D5D4DJD2H9H4HAH3DTD6D3D9D8S6D7HTDQC5DAC4CKCAC2C6STH6S8C3S7H7S9C7SJH8C8CTHQCJHJHKDKH5CQC9
W:A653.987.8752.74 942.T32.Q43.AKJ3 KQJ.KQ6.KJ6.QT98 T87.AJ54.AT9.652
E EW 1N.-1.E
1N PP PP PP
H4H7HTHKSKS8S3S9SJS7S5S2SQSTSAS4S6H2C9C6C4CKC8C5H3HQHAH8HJH9D3H6H5D2C3D6C2C7CACTCJCQD9D5DJDTD7DQD4DKDAD8
...
```
Run the script to transform the data into binary format. A single script now covers all four seats, and splits each of them into a suit variant and a NT variant:

```
python binary_keras_suit_nt.py [input.txt ...] [out_dir=./binary] [workers=4]
```

Without input files it reads `../../data/WBC/play.txt`. The eight arrays are built in separate processes,
`workers` at a time; a suit array of 1.2 million deals takes 8 GB while it is built, so four at a time need
about 32 GB (`workers=1` builds them one after the other).

The WBC file is not in the repository - it is published on the
[training-data release](https://github.com/lorserker/ben/releases/tag/training-data) as
[play_jackformat.zip](https://github.com/lorserker/ben/releases/download/training-data/play_jackformat.zip)
(the same boards in BEN format are in [play_ben.zip](https://github.com/lorserker/ben/releases/download/training-data/play_ben.zip)).

**The declarer letter in the published WBC file is wrong in three records out of four** (the dealer letter and
the auction give the right declarer in 99.8% of them). The converter therefore does not use the letter: the
declarer is the right-hand opponent of the hand that holds the card led to the first trick, and records whose
play does not fit the hands (0.04%) are left out. `check_play_data.py` reports this for any play file:

```
python check_play_data.py ../../data/WBC/play.txt
```

Arrays made from the WBC file by an older converter show three deals out of four from the wrong seats; a model
scores about 33% on such an array against 77% on a right one, which is how to tell them apart.

This writes 16 files into the `binary` folder - `X_<seat>_<suit|nt>.npy` and `Y_<seat>_<suit|nt>.npy` for each of `lefty`, `righty`, `decl` and `dummy`. `X` contains the inputs to the neural network and `Y` the expected outputs, both in numpy array format.

Then, run the trainig script. The network is a 3-layer LSTM over the tricks, so the card played can depend on everything seen earlier in the play (signals, show-outs, suits already played), not only on the last trick. The models need Keras 3.15 or later to load.

Use a GPU (on Windows that means WSL): on an RTX 4090 an epoch over the full data takes well under a minute. The last 5% of the deals are held out for validation, training stops when the validation loss stops improving, and a snapshot is saved after every epoch.

```
python play_nn_keras.py binary decl_nt
python play_nn_keras.py binary decl_suit
python play_nn_keras.py binary dummy_nt
python play_nn_keras.py binary dummy_suit
python play_nn_keras.py binary lefty_nt
python play_nn_keras.py binary lefty_suit
python play_nn_keras.py binary righty_nt
python play_nn_keras.py binary righty_suit
```

The first argument is the directory holding the binary files, the second selects which of them to train on. Options can follow as `key=value`: `lstm_size` (128), `n_layers` (3), `dense_layers` (0), `dense_units` (512), `epochs` (30), `batch_size` (1024), `learning_rate` (0.001), `keep` (0.8), `val_fraction` (0.05), `patience` (3), `max_examples` (0 = all), `name`, `schedule` (`plateau`), `final_lr` (0.02), `gpu_data` (1) and `seed` (0 = not set). The generated model is named after that selection plus the date, and is written to the `model` folder; `<name>_<date>-best.keras` is the snapshot with the best validation loss.

With a GPU the training deals are held on the GPU in compact form and several hundred training steps run as one program (`../gpu_epochs.py`), with the LSTM layers unrolled over the tricks and compiled by XLA: an epoch of `decl_suit` (1.2 million deals, the hybrid network below) takes 10 seconds on an RTX 5080 against 34 seconds with `model.fit()` (2026-11-18, nothing else on the GPU); the first epoch takes about a minute longer while the program is compiled. `gpu_data=0` selects `model.fit()`; it is also used without a GPU and for arrays that are not of the expected form. The saved models are the same either way.

`schedule=plateau` (the default) multiplies the learning rate by 0.3 when the validation loss stops improving and stops after `patience` epochs without improvement. `schedule=cosine` lowers the learning rate along half a cosine to `final_lr` times its start over all epochs, without an early stop; it scored worse than `plateau` in every run of November 2026.

The shipped `*_hybrid_2026-10-16.keras` models have two dense layers of 512 before two LSTM layers of 256. With `epochs=30` that network is still improving when training ends; with less dropout and room to finish it scores better on the held-out deals (see `../pipeline/README.md` for the runs):

```
python play_nn_keras.py binary decl_suit dense_layers=2 dense_units=512 lstm_size=256 n_layers=2 keep=0.9 epochs=60
```

For the four NT seats, which have half as many deals, `keep=0.8` is better than `keep=0.9`.

To compare models on the held-out deals:

```
python eval_play_nn.py binary righty_suit ../../../models/TF2models/righty_suit_2024-07-16-E20.keras model/righty_suit_<date>-best.keras
```

It prints, per trick and in total, how often the model's top card is the card actually played, and `nll` - how unlikely the model found the card actually played. `nll` is what matters when BEN uses the model to score samples.

When the network is completed, you can plug it back into the engine to use instead of the default one it came with. To do that, set the path to the network which you just trained in a configuration file - either `src/config/default.conf` or a new configuration file of your own.

The neural network for leads are using the same principles, but is generating more datasets.

### Making a test run

To test the neural network, it is possible to run game.py and see how the play proceeds

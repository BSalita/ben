import sys
import datetime
import os
import time
import numpy as np

# Set logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import gpu_epochs

# Set TensorFlow to only allocate as much GPU memory as needed
physical_devices = tf.config.list_physical_devices('GPU')
print("physical_devices", physical_devices)
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Using GPU: ", physical_devices)
else:
    print("Using CPU only")

# Check for correct usage
if len(sys.argv) < 3:
    print("Usage: python play_nn_keras.py inputdirectory hand [key=value ...]")
    print("  hand: decl_nt, decl_suit, dummy_nt, dummy_suit, lefty_nt, lefty_suit, righty_nt, righty_suit")
    print("  options: lstm_size=128 n_layers=3 dense_layers=0 dense_units=512 epochs=30 batch_size=1024 learning_rate=0.001")
    print("           keep=0.8 val_fraction=0.05 patience=3 max_examples=0 name=<hand>")
    print("           schedule=plateau|cosine final_lr=0.02 gpu_data=1 seed=0")
    print("  schedule=plateau: learning rate times 0.3 when the validation loss stops improving, and")
    print("           training stops after <patience> epochs without improvement (the default)")
    print("  schedule=cosine: half a cosine from learning_rate to final_lr times it over all epochs;")
    print("           no early stop, the model after the last epoch is saved")
    print("  gpu_data=0: feed batches from memory with model.fit() even when a GPU is present")
    sys.exit(1)

bin_dir = sys.argv[1]
hand = sys.argv[2]
print(sys.argv)

options = dict(arg.split('=', 1) for arg in sys.argv[3:])
lstm_size = int(options.get('lstm_size', 128))
n_layers = int(options.get('n_layers', 3))
# Optional Dense layers per trick before the LSTM layers
dense_layers = int(options.get('dense_layers', 0))
dense_units = int(options.get('dense_units', 512))
epochs = int(options.get('epochs', 30))
batch_size = int(options.get("batch_size", 1024))
learning_rate = float(options.get('learning_rate', 0.001))
keep = float(options.get('keep', 0.8))
# The last deals in the file are held out for validation; eval_play_nn.py uses the same split
val_fraction = float(options.get('val_fraction', 0.05))
patience = int(options.get('patience', 3))
max_examples = int(options.get('max_examples', 0))
name = options.get('name', hand)
schedule = options.get('schedule', 'plateau')
final_lr = float(options.get('final_lr', 0.02))
seed = int(options.get('seed', 0))
if seed:
    keras.utils.set_random_seed(seed)

X_all = np.load(os.path.join(bin_dir, f'X_{hand}.npy'), mmap_mode='r')
y_all = np.load(os.path.join(bin_dir, f'Y_{hand}.npy'), mmap_mode='r')
if max_examples:
    X_all, y_all = X_all[:max_examples], y_all[:max_examples]

n_all = X_all.shape[0]
n_val = int(n_all * val_fraction)
n_train = n_all - n_val
n_tricks = X_all.shape[1]
n_ftrs = X_all.shape[2]
n_card = y_all.shape[2]

# With a GPU the training deals are held on it in compact form and several hundred training
# steps run as one program (see gpu_epochs.py). Otherwise, or when the arrays are not of the
# expected form, they are loaded into memory for model.fit().
t = time.time()
gpu_data = None
if physical_devices and int(options.get('gpu_data', 1)):
    print("Preparing the training data for the GPU")
    compact_x = gpu_epochs.compact_play(X_all[:n_train], n_cards=n_card)
    compact_y = gpu_epochs.compact_card_targets(y_all[:n_train])
    if compact_x is not None and compact_y is not None:
        gpu_data = compact_x + (compact_y,)
    print("Training data on the GPU:", gpu_data is not None)
if gpu_data is None:
    print("Loading data into memory")
    X_train, y_train = np.asarray(X_all[:n_train]), np.asarray(y_all[:n_train])
X_val, y_val = np.asarray(X_all[n_train:]), np.asarray(y_all[n_train:])
print(f"Loaded in {time.time() - t:0.1f} seconds")

steps_per_epoch = n_train // batch_size
model_name = f'{name}_{datetime.datetime.now().strftime("%Y-%m-%d")}'

print("-------------------------")
print("Examples for training:   ", n_train)
print("Examples for validation: ", n_val)
print("Model path:              ", model_name)
print("-------------------------")
print("Tricks per example:      ", X_all.shape[1])
print("Size input:              ", n_ftrs)
print("Size output card:        ", n_card)
print("-------------------------")
print("Dense layers:            ", dense_layers, "x", dense_units)
print("LSTM size:               ", lstm_size)
print("Layers:                  ", n_layers)
print("Batch size:              ", batch_size)
print("steps_per_epoch          ", steps_per_epoch)
print("Learning rate:           ", learning_rate)
print("Schedule:                ", schedule)
print("Keep:                    ", keep)

learning_rate_schedule = learning_rate
if schedule == 'cosine':
    learning_rate_schedule = optimizers.schedules.CosineDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=steps_per_epoch * epochs, alpha=final_lr)


def batches(X, y, batch_size, shuffle):
    """Batches gathered from the in-memory arrays, prepared in parallel with training."""
    with tf.device('/CPU:0'):
        X_t, y_t = tf.constant(X), tf.constant(y)
    ds = tf.data.Dataset.range(len(X))
    if shuffle:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.map(lambda i: (tf.gather(X_t, i), tf.gather(y_t, i)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    # A recurrent model, so the card played can depend on everything seen earlier in
    # the play (signals, show-outs, suits played), not only on the last trick.
    # The output is a plain Dense layer: it works per trick on the sequence, and
    # TimeDistributed is broken in Keras 3.15 for an unknown number of tricks.
    inputs = keras.Input(shape=(None, n_ftrs), dtype='float16', name='x_play')
    x = inputs
    for _ in range(dense_layers):
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(1 - keep)(x)
    for _ in range(n_layers):
        x = layers.LSTM(lstm_size, return_sequences=True)(x)
        x = layers.Dropout(1 - keep)(x)
    card = layers.Dense(n_card, activation='softmax', name='card')(x)
    model = models.Model(inputs=inputs, outputs=card)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


checkpoint_dir = "model"
os.makedirs(checkpoint_dir, exist_ok=True)

# Continue from the latest checkpoint for this name, if there is one
initial_epoch = 0
checkpoints = sorted((int(f[len(model_name) + 2:-len('.keras')]), f) for f in os.listdir(checkpoint_dir)
                     if f.startswith(model_name + '-E') and f.endswith('.keras') and f[len(model_name) + 2:-len('.keras')].isdigit())
if checkpoints:
    initial_epoch, latest = checkpoints[-1]
    model_path = os.path.join(checkpoint_dir, latest)
    print(f"Loading saved model from {model_path}, continuing after epoch {initial_epoch}")
    model = keras.models.load_model(model_path)
else:
    model = build_model()
model.summary()

if initial_epoch >= epochs:
    print("Model training complete after", initial_epoch, "epochs")
    sys.exit(0)


class EpochTimer(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch took {time.time() - self.t_start:0.1f} seconds')


# On the GPU the LSTM layers are unrolled over the tricks and the training step is compiled by
# XLA, which is several times faster than the cuDNN LSTM kernel for networks of this size
# (see bidding_nn_keras.py). The weights are the same either way; the layers are switched
# back while saving, so a saved model still takes any number of tricks.
lstm_layers = [layer for layer in model.layers if isinstance(layer, layers.LSTM)] if gpu_data is not None else []


def set_unroll(value):
    for layer in lstm_layers:
        layer.unroll = value
        layer.supports_jit = value


def save_model(path):
    set_unroll(False)
    try:
        model.save(path)
    finally:
        set_unroll(True)


class Checkpoint(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_model(os.path.join(checkpoint_dir, f"{model_name}-E{epoch + 1:02d}.keras"))


training_callbacks = [Checkpoint()]
if schedule != 'cosine':
    training_callbacks += [
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
    ]
training_callbacks += [callbacks.CSVLogger(os.path.join(checkpoint_dir, f"{model_name}.csv"), append=True), EpochTimer()]

print("Training started")
if gpu_data is not None:
    set_unroll(True)
    model.jit_compile = True
    print("LSTM layers unrolled for training:", len(lstm_layers), "XLA:", model.jit_compile)
    with tf.device('/GPU:0'):
        g_before, g_cards, g_after, g_y = (tf.constant(a) for a in gpu_data)
    n_blocks = gpu_data[1].shape[2]

    def batch(idx):
        cards = tf.one_hot(tf.gather(g_cards, idx), n_card, dtype=tf.float16)
        x = tf.concat([tf.cast(tf.gather(g_before, idx), tf.float16),
                       tf.reshape(cards, (batch_size, n_tricks, n_blocks * n_card)),
                       tf.cast(tf.gather(g_after, idx), tf.float16)], axis=2)
        return x, tf.one_hot(tf.gather(g_y, idx), n_card, dtype=tf.float16)

    def validation():
        logs = model.evaluate(X_val, y_val, batch_size=4096, verbose=0, return_dict=True)
        return {'val_' + k: v for k, v in logs.items()}

    gpu_epochs.fit_batches(model, batch, n_train, batch_size, epochs, steps_per_epoch, callbacks=training_callbacks,
                           validation=validation, initial_epoch=initial_epoch)
else:
    train_batches = batches(X_train, y_train, batch_size, shuffle=True)
    val_batches = batches(X_val, y_val, 1024, shuffle=False)
    model.fit(train_batches, validation_data=val_batches,
              epochs=epochs, initial_epoch=initial_epoch, callbacks=training_callbacks)

final_model_path = os.path.join(checkpoint_dir, f"{model_name}-best.keras")
save_model(final_model_path)
print("Saved model with the best validation loss:" if schedule != 'cosine' else "Saved model:", final_model_path)

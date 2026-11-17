"""Turn the weights written by bidding_nn_torch.py into the .keras model BEN loads.

Usage: python torch_to_keras.py weights.npz output.keras

Builds the same network as bidding_nn_keras.py (net=bidding) or
../bidding_info/binfo_nn_keras.py (net=info) and sets its weights. Needs TensorFlow, not PyTorch.
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

data = np.load(sys.argv[1])
net = str(data['net'])
lstm_size, n_layers, n_ftrs = int(data['lstm_size']), int(data['n_layers']), int(data['n_ftrs'])
weights = [data[k] for k in sorted(k for k in data.files if k.startswith('w'))]

inputs = tf.keras.Input(shape=(None, n_ftrs), dtype=tf.float16)
x = inputs
for _ in range(n_layers):
    x = layers.LSTM(lstm_size, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
if net == 'bidding':
    outputs = layers.TimeDistributed(layers.Dense(weights[-1].shape[0], activation='softmax'), name='bid_output')(x)
else:
    outputs = [layers.TimeDistributed(layers.Dense(weights[-3].shape[0], activation='linear'), name='hcp_output')(x),
               layers.TimeDistributed(layers.Dense(weights[-1].shape[0], activation='linear'), name='shape_output')(x)]
model = models.Model(inputs=inputs, outputs=outputs)
model.set_weights(weights)
model.save(sys.argv[2])
print("Saved", sys.argv[2], "from", sys.argv[1], f"({net}, {len(weights)} weight arrays)")

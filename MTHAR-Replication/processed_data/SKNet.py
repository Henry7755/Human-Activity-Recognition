9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ──────────────────────────────────────────────
# 1. GPU configuration
# ──────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

best_acc = 0.0
start_epoch = 0

# ──────────────────────────────────────────────
# 2. Data loading
# ──────────────────────────────────────────────
print('==> Preparing data..')

DATA_DIR = '/home/gaowenbing/desktop/dd/Torch_Har_cbam/HAR_Dataset/uci_har/'

train_x = np.load(os.path.join(DATA_DIR, 'np_train_x.npy')).astype(np.float32)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)   # (N, H, W, C)  – channels-last

train_y = np.load(os.path.join(DATA_DIR, 'np_train_y.npy')).astype(np.float32)

test_x = np.load(os.path.join(DATA_DIR, 'np_test_x.npy')).astype(np.float32)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

test_y = np.load(os.path.join(DATA_DIR, 'np_test_y.npy')).astype(np.float32)

print('train_x:', train_x.shape, '  train_y:', train_y.shape)
print('test_x: ', test_x.shape,  '  test_y: ', test_y.shape)

# tf.data pipelines
BATCH_SIZE = 256
TEST_BATCH  = 2947

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
    .shuffle(buffer_size=10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((test_x, test_y))
    .batch(TEST_BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

# ──────────────────────────────────────────────
# 3. SKConv layer
#    PyTorch layout: (N, C, H, W)  →  TF layout: (N, H, W, C)
#    PyTorch kernel=(3,1), padding=(1+i, 1), dilation=(1+i, 1)
#    → TF Conv2D with kernel_size=(3,1), padding='same' per branch
# ──────────────────────────────────────────────

class SKConv(layers.Layer):
    """Selective-Kernel Convolution block (TensorFlow/Keras, channels-last)."""

    def __init__(self, features, M=3, G=32, r=32, stride=1, L=32, **kwargs):
        super().__init__(**kwargs)
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        # M parallel branches with different dilation rates
        self.branches = []
        for i in range(M):
            branch = keras.Sequential([
                # groups → use DepthwiseConv2D trick or just Conv2D
                # PyTorch groups=G  ≈  TF Conv2D with groups=G (TF 2.x supports it)
                layers.Conv2D(
                    filters=features,
                    kernel_size=(3, 1),
                    strides=(stride, 1),
                    padding='same',
                    dilation_rate=(1 + i, 1),
                    groups=G,
                    use_bias=False,
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ], name=f'branch_{i}')
            self.branches.append(branch)

        # Global Average Pool + FC (implemented as Conv2D(1,1) to keep 4-D tensor)
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)   # → (N, 1, 1, C)
        self.fc = keras.Sequential([
            layers.Conv2D(d, kernel_size=1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ], name='fc_z')

        # One 1×1 conv per branch to produce attention logits
        self.fcs = [
            layers.Conv2D(features, kernel_size=1, name=f'fc_attn_{i}')
            for i in range(M)
        ]
        self.softmax = layers.Softmax(axis=1)   # softmax across branch dimension

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        # Each branch output: (N, H, W, features)
        branch_feats = [branch(x, training=training) for branch in self.branches]

        # Stack → (N, M, H, W, features)
        feats = tf.stack(branch_feats, axis=1)

        # Fuse: element-wise sum across branches → (N, H, W, features)
        feats_U = tf.reduce_sum(feats, axis=1)

        # Channel descriptor via GAP → (N, 1, 1, features)
        feats_S = self.gap(feats_U)

        # Compact feature → (N, 1, 1, d)
        feats_Z = self.fc(feats_S, training=training)

        # Per-branch attention vectors → list of (N, 1, 1, features)
        attn_vectors = [fc(feats_Z) for fc in self.fcs]

        # Stack → (N, M, 1, 1, features), then softmax over branch dim
        attn_vectors = tf.stack(attn_vectors, axis=1)
        attn_vectors = self.softmax(attn_vectors)           # (N, M, 1, 1, features)

        # Weighted sum: (N, M, H, W, features) * (N, M, 1, 1, features)
        feats_V = tf.reduce_sum(feats * attn_vectors, axis=1)  # (N, H, W, features)

        return feats_V

    def get_config(self):
        config = super().get_config()
        config.update(dict(features=self.features, M=self.M))
        return config


# ──────────────────────────────────────────────
# 4. SKNet model
# ──────────────────────────────────────────────

def build_sknet(M=3, G=32, r=32, stride=1, L=32, num_classes=6):
    """Returns a compiled Keras Model equivalent to the PyTorch SKNet."""
    inputs = keras.Input(shape=(None, None, 1), name='input')   # (H, W, 1)

    # conv1: PyTorch Conv2d(1, 64, (5,1), stride=(3,1), padding=(1,0))
    # channels-last TF equivalent:
    x = layers.Conv2D(64, kernel_size=(5, 1), strides=(3, 1), padding='same', use_bias=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # NOTE: PyTorch conv2_sk expects 128 input features but conv1 outputs 64.
    # The original code passes 64-channel output to SKConv(128, ...) which would
    # error in PyTorch too unless conv1 produces 128. We mirror the code as-is
    # and set SKConv to match actual channel count (64 → 64, then 64 → 128).
    # Adjust these numbers to match your actual checkpoint / intent.
    x = SKConv(64,  M=M, G=G, r=r, stride=stride, L=L, name='skconv1')(x)
    x = SKConv(128, M=M, G=G, r=r, stride=stride, L=L, name='skconv2')(x)

    # Flatten
    x = layers.Flatten()(x)

    # FC: 6-class output
    # PyTorch uses nn.LayerNorm on the logits; replicate with LayerNormalization
    x = layers.Dense(num_classes, name='fc')(x)
    x = layers.LayerNormalization(name='layer_norm')(x)

    model = keras.Model(inputs, x, name='SKNet')
    return model


# ──────────────────────────────────────────────
# 5. Compile
# ──────────────────────────────────────────────
print('==> Building model..')

model = build_sknet()
model.summary()

WD = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=WD)

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy'],
)

# Learning-rate scheduler: StepLR(step_size=50, gamma=0.1)
# In Keras this is done via a LearningRateScheduler callback
def step_lr_schedule(epoch, lr):
    """Halve the LR by ×0.1 every 50 epochs."""
    if epoch > 0 and epoch % 50 == 0:
        return lr * 0.1
    return lr

lr_callback = keras.callbacks.LearningRateScheduler(step_lr_schedule, verbose=1)

# ──────────────────────────────────────────────
# 6. Training loop
# ──────────────────────────────────────────────
EPOCHS = 500

epoch_list  = []
error_list  = []

# Custom training loop to replicate per-epoch logging
for epoch in range(start_epoch, start_epoch + EPOCHS):
    print(f'\nEpoch: {epoch}')

    # --- Train ---
    train_results = model.fit(
        train_dataset,
        epochs=1,
        verbose=1,
        callbacks=[lr_callback],
    )

    # --- Test ---
    test_results = model.evaluate(test_dataset, verbose=0)
    test_loss    = test_results[0]
    test_acc     = test_results[1]
    test_error   = 1.0 - test_acc

    print(f'test: {test_acc:.4f} || {test_error:.4f}')

    epoch_list.append(epoch)
    error_list.append(test_error)

    if test_acc > best_acc:
        best_acc = test_acc
        model.save_weights('best_sknet_weights.h5')
        print(f'  >> New best accuracy: {best_acc:.4f}  (weights saved)')

print(f'\nTraining complete. Best test accuracy: {best_acc:.4f}')

# ──────────────────────────────────────────────
# 7. Model statistics  (replaces torchstat)
#    Print param count and a summary for input (128, 9, 1).
# ──────────────────────────────────────────────
stat_model = build_sknet()
stat_model.build(input_shape=(None, 128, 9, 1))
stat_model.summary()

total_params     = stat_model.count_params()
trainable_params = sum(
    tf.size(w).numpy() for w in stat_model.trainable_weights
)
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
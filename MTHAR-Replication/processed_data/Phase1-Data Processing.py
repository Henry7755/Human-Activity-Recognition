# %%
import numpy as np
import pandas as pd

# %%
df_ytrain = pd.read_csv("/Human Activity Recognition/MTHAR-Replication/processed_data/y_train.txt", sep =r"\s+", header=None)
df_Xtrain = pd.read_csv("/Human Activity Recognition/MTHAR-Replication/processed_data/X_train.txt", sep =r"\s+", header=None)

# %%
# This represents the features and labels for the training set, it cannot be classified as time series data yet.
print(df_Xtrain.shape)
print(df_ytrain.shape)


# %% [markdown]
# The structuring of the data for easy training. Since we are using Selective_kernel Network, which is CNN based for Human Activity Recognition and Segmentation
# 
# The first step is the processing of the data to be used for training. From Better Deep Learning it is required that you use 3D of getting the data ready.
# The 3D structure of the input data is often summarized using the arrary shape of the notation [samples, timesteps, features]. The 2D format of this should be [samples, features]

# %%

def load_raw_signals(folder_path, filenames):
    signals = []
    for name in filenames:
        # Each file is (7352, 128)
        data = pd.read_csv(f'{folder_path}/{name}.txt', header=None, delim_whitespace=True)
        signals.append(data.values)
    
    # Stacking on the 3rd axis (axis=2) results in (7352, 128, 9)
    return np.transpose(np.array(signals), (1, 2, 0))

# Example list of the 9 channels based on your files
filenames = [
    'total_acc_x_train', 'total_acc_y_train', 'total_acc_z_train',
    'body_acc_x_train', 'body_acc_y_train', 'body_acc_z_train',
    'body_gyro_x_train', 'body_gyro_y_train', 'body_gyro_z_train'
]

X_train = load_raw_signals('/home/pschye/Documents/KCCR - Biosignals/Human Activity Recognition/Human Activity Recognition/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals', filenames)

# %%
X_train.shape

# %% [markdown]
# (7352, 128, 9) this is a workable format to be used in CNN(Backbone Network)

# %% [markdown]
# ## The Method in replicating the MTHAR work 
# To reconstruct the continuous time-series stream from the pre-segmented UCI HAR dataset according to the methodology in the Duan et al. (2023) paper, you need to "stitch" the overlapping windows back together.
# 
# The UCI HAR dataset was originally segmented into windows of 128 samples with a 50% overlap (64 samples)2. Because the paper focuses on activity segmentation (identifying starting and ending positions), reconstructing the full stream is necessary to define the ground truth boundaries $(t^x, l^x)$ for their multi-task learning model
# 
# 1. Grouping by Subject
# The first step is to isolate data for each individual participant.Use the subject_train.txt file, which contains a subject ID (1–30) for every row (window) in your feature matrix.Filter your raw signal tensor—shaped (7352, 128, 9)—so that you only process windows belonging to one subject at a time.
# 
# 2. The Reconstruction (Stitching) Process
# Since each window starts 64 samples after the previous one, you must blend the overlapping regions. The paper implies a "smoothing" approach through averaging to handle the transitions between windows.
# 
# Logic for one channel of a single subject:Initialize: Create an empty array for the reconstructed signal and a "counter" array of the same length to track how many windows cover each index.
# Total Length = (Number of Windows * 64) + 648.
# Accumulate: Loop through each window $i$:Place the 128 samples into the reconstructed array starting at index $i \times 64$.
# Add 1 to the corresponding indices in your counter array.
# Average: Divide the total accumulated signal by the counter array.Indices 0–63 and the final 64 samples will be divided by 1.Indices 64 to (Total Length - 64) will be divided by 2 (because they represent the overlap)
# 
# 3. Reconstructing Ground Truth Labels
# The y_train file provides one label per window. To replicate the researchers' "Ground Truth Boundaries," you must map these back to the samples:Assign Labels: 
# Assign the label of Window $i$ to the 64-sample "step" starting at $i \times 64$.
# Define Boundaries: Scan the resulting sample-level label sequence.
#  A boundary is identified whenever the activity label changes (e.g., from WALKING to SITTING).Center and Length $(t^x, t^l)$: Convert these segments into the paper's required format:$t^x$: The center coordinate of the activity segment15.$t^l$: The total number of samples (length) of that activity
#  
#  4. Summary for the 9 ChannelsRepeat this process for all 9 channels (X/Y/Z for Total Acc, Body Acc, and Gyro). 
#  You will end up with a continuous multivariate stream of shape (Total_Samples, 9) for each subject, which the MTHARS model uses to predict both the activity class and the offset to the true bound
# 

# %%

def reconstruct_har_data(X_windows, y_windows, subject_ids):
    """
    Reconstructs continuous signals and activity boundaries per subject.
    X_windows: (7352, 128, 9) raw signal tensor
    y_windows: (7352,) activity labels
    subject_ids: (7352,) subject IDs from subject_train.txt
    """
    unique_subjects = np.unique(subject_ids)
    all_reconstructed_data = {}

    for sub in unique_subjects:
        # 1. Filter data for the specific subject
        # Use .reshape(-1) to ensure the mask is a flat 1D array
        mask = (subject_ids.reshape(-1) == sub) 

# Explicitly filter the first axis
        sub_X = X_windows[mask, :, :]  
        sub_y = y_windows[mask]
        
        num_windows = sub_X.shape[0]
        overlap = 64 # 50% overlap of 128 samples
        total_len = (num_windows * overlap) + overlap
        
        # 2. Initialize reconstructed signal and count array for averaging
        sub_stream = np.zeros((total_len, 9))
        counts = np.zeros((total_len, 1))
        
        # 3. Stitch windows with 50% overlap
        for i in range(num_windows):
            start = i * overlap
            end = start + 128
            sub_stream[start:end, :] += sub_X[i]
            counts[start:end] += 1
            
        # 4. Average overlapping sections to smooth the signal
        sub_stream /= counts
        
        # 5. Reconstruct label sequence (propagate window label to its 64-sample step)
        sub_labels = np.zeros(total_len)
        for i in range(num_windows):
            sub_labels[i*overlap : (i+1)*overlap] = sub_y[i]
        # Fill the final 64 samples with the last known label
        sub_labels[-64:] = sub_y[-1]
        
        # 6. Extract Boundaries (tx: center, tl: length) as per Duan et al.
        boundaries = []
        # Find indices where labels change
        change_points = np.where(np.diff(sub_labels) != 0)[0]
        start_idx = 0
        
        for cp in change_points:
            end_idx = cp
            length = end_idx - start_idx + 1
            center = start_idx + (length / 2)
            boundaries.append({'activity': sub_labels[start_idx], 'tx': center, 'tl': length})
            start_idx = end_idx + 1
            
        all_reconstructed_data[sub] = {
            'signal': sub_stream, 
            'labels': sub_labels,
            'boundaries': boundaries
        }
        
    return all_reconstructed_data

# --- Usage Example ---
# Assuming X_train_raw is (7352, 128, 9)
# y_train is (7352,)
# sub_train is (7352,) from subject_train.txt
# reconstructed = reconstruct_har_data(X_train_raw, y_train, sub_train)

# %%
y_train  = np.loadtxt("/home/pschye/Documents/KCCR - Biosignals/Human Activity Recognition/Human Activity Recognition/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt")
subjects = np.loadtxt("/home/pschye/Documents/KCCR - Biosignals/Human Activity Recognition/Human Activity Recognition/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/subject_train.txt")

# %%
# Apply the function to your data
reconstructed_data = reconstruct_har_data(X_train, y_train, subjects)

# Example: Accessing the continuous data for Subject 3
sub1_signal = reconstructed_data[3]['signal']   # Continuous (Length, 9) stream
sub1_labels = reconstructed_data[3]['labels']   # Sample-level activity labels
sub1_events = reconstructed_data[3]['boundaries'] # List of (tx, tl) events

# %%
print(len(sub1_signal))

# %% [markdown]
# # MTHAR - THE METHOD IN ACTION USING TENSORFLOW
# 

# %%


# --- Phase 2: Anchor Configuration ---
SCALES = [0.2, 0.4, 0.6, 0.8, 1.0]
BASE_WINDOW = 128  # n in the paper
STRIDE = 8         # Downsampling factor of the backbone

def get_anchors(signal_len):
    """
    Generates anchor windows (center, width) across the signal.
    """
    anchors = []
    # We generate anchors centered on every 'unit' of the feature map
    for center in range(0, signal_len, STRIDE):
        for s in SCALES:
            # Two widths per scale as per paper formula
            w1 = BASE_WINDOW * np.sqrt(s)
            w2 = BASE_WINDOW / np.sqrt(s)
            anchors.append([center, w1])
            anchors.append([center, w2])
    return np.array(anchors)

# --- Phase 3: Multi-Task Model (Recognition + Localization) ---

class MTHARS_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(MTHARS_Model, self).__init__()
        # Backbone (From your SKNet implementation)
        self.init_conv = models.Sequential([
            layers.Conv2D(64, (5, 1), strides=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.sk1 = SKConv(128) # Your previous SKConv class
        self.sk2 = SKConv(256)
        
        # We assume 10 anchors per feature map location (5 scales * 2 widths)
        self.num_anchors = len(SCALES) * 2
        
        # Head 1: Recognition (Classification)
        # Output shape: [Batch, Feature_Map_Len, Num_Anchors * Num_Classes]
        self.cls_head = layers.Conv2D(self.num_anchors * num_classes, (3, 1), padding='same')
        
        # Head 2: Localization (Regression for offsets dx, dl)
        # Output shape: [Batch, Feature_Map_Len, Num_Anchors * 2]
        self.loc_head = layers.Conv2D(self.num_anchors * 2, (3, 1), padding='same')

    def call(self, x):
        features = self.init_conv(x)
        features = self.sk1(features)
        features = self.sk2(features) # Shape: [Batch, H, W, 256]
        
        # Predict Class Confidences
        cls_preds = self.cls_head(features)
        # Reshape to [Batch, Total_Anchors, Num_Classes]
        cls_preds = tf.reshape(cls_preds, (tf.shape(x)[0], -1, 6))
        
        # Predict Boundary Offsets (fx, fl)
        loc_preds = self.loc_head(features)
        # Reshape to [Batch, Total_Anchors, 2]
        loc_preds = tf.reshape(loc_preds, (tf.shape(x)[0], -1, 2))
        
        return cls_preds, loc_preds

# Initialize Model
model = MTHARS_Model(num_classes=6)

# %%
import tensorflow as tf
from tensorflow.keras import layers, models

class SKConv(layers.Layer):
    def __init__(self, features, M=3, G=32, r=32, stride=1, L=32):
        super(SKConv, self).__init__()
        self.M = M
        self.features = features
        self.d = max(int(features / r), L)
        
        # Branching Convolutions (Split)
        self.convs = []
        for i in range(M):
            # Using dilation to mimic different receptive fields as per the paper
            self.convs.append(models.Sequential([
                layers.Conv2D(features, kernel_size=(3, 1), strides=stride, 
                              padding='same', dilation_rate=(1 + i, 1), groups=G, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()
            ]))

        self.gap = layers.GlobalAveragePooling2D()
        
        # Attention bottleneck (Fuse)
        self.fc = models.Sequential([
            layers.Dense(self.d, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        # Selection heads
        self.fcs = [layers.Dense(features) for _ in range(M)]
        self.softmax = layers.Softmax(axis=1)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        
        # Split
        feats = [conv(x) for conv in self.convs] # List of [batch, h, w, c]
        feats_stack = tf.stack(feats, axis=1)    # [batch, M, h, w, c]
        
        # Fuse
        feats_U = tf.reduce_sum(feats_stack, axis=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        
        # Select
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = tf.stack(attention_vectors, axis=1) # [batch, M, c]
        attention_vectors = self.softmax(attention_vectors)
        
        # Reshape for broadcasting: [batch, M, 1, 1, c]
        attention_vectors = tf.expand_dims(tf.expand_dims(attention_vectors, axis=2), axis=2)
        
        # Weighted sum
        feats_V = tf.reduce_sum(feats_stack * attention_vectors, axis=1)
        return feats_V

def build_mthars_model(input_shape=(128, 9, 1), num_classes=6):
    inputs = layers.Input(shape=input_shape)

    # --- Backbone Network ---
    # Initial Conv Block
    x = layers.Conv2D(64, (5, 1), strides=(3, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # SKConv Blocks
    x = SKConv(128)(x)
    x = SKConv(256)(x)
    
    # Flatten/Dense (Transition to Recognition/Segmentation Network)
    # The paper uses a "Windows Generate" and Conv1D structure here
    # For a direct conversion of your snippet:
    x_flatten = layers.Flatten()(x)
    
    # --- Multi-Task Heads (Based on Paper Figure 3) ---
    
    # 1. Recognition Head (Activity Category)
    rec_head = layers.Dense(num_classes, name='recognition_output')(x_flatten)
    rec_head = layers.LayerNormalization()(rec_head)
    rec_output = layers.Activation('softmax')(rec_head)
    
    # 2. Segmentation Head (Activity vs Transition)
    seg_head = layers.Dense(2, name='segmentation_output')(x_flatten)
    seg_head = layers.LayerNormalization()(seg_head)
    seg_output = layers.Activation('softmax')(seg_head)

    model = models.Model(inputs=inputs, outputs=[rec_output, seg_output])
    return model

# Initialize and Compile
model = build_mthars_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'recognition_output': 'categorical_crossentropy',
        'segmentation_output': 'categorical_crossentropy'
    },
    metrics=['accuracy']
)

model.summary()



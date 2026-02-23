import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Base directory for the dataset
base_dir = '/home/pschye/Documents/KCCR - Biosignals/Human Activity Recognition/Human Activity Recognition/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset'

# Define the 9 channels
channels = [
    'body_acc_x', 'body_acc_y', 'body_acc_z',
    'total_acc_x', 'total_acc_y', 'total_acc_z',
    'body_gyro_x', 'body_gyro_y', 'body_gyro_z'
]

# Function to reconstruct streams for train or test
def reconstruct_streams(split='train'):
    # Load subjects and labels
    subjects = np.loadtxt(f'{base_dir}/{split}/subject_{split}.txt', dtype=int)
    y = np.loadtxt(f'{base_dir}/{split}/y_{split}.txt', dtype=int)
    
    # Load all channel data: dict of (num_windows, 128)
    data = {ch: np.loadtxt(f'{base_dir}/{split}/Inertial Signals/{ch}_{split}.txt') for ch in channels}
    
    unique_subjects = np.unique(subjects)
    full_signals = {}
    sample_labels = {}
    boundaries = {}  # Per subject: list of (start, end, label) tuples
    
    for sub in unique_subjects:
        idx = np.where(subjects == sub)[0]
        num_windows = len(idx)
        signal_len = (num_windows - 1) * 64 + 128  # Account for 50% overlap
        signal = np.zeros((signal_len, 9))
        labels = np.zeros(signal_len, dtype=int)
        
        for i, win_idx in enumerate(idx):
            start = i * 64
            # Stack channels for this window: (128, 9)
            win_data = np.column_stack([data[ch][win_idx] for ch in channels])
            
            if i == 0:
                signal[start:start+128] = win_data
            else:
                # Average overlap
                signal[start:start+64] = (signal[start:start+64] + win_data[:64]) / 2
                signal[start+64:start+128] = win_data[64:]
            
            # Assign label (use current window's label for overlap)
            labels[start:start+128] = y[win_idx]
        
        full_signals[sub] = signal
        sample_labels[sub] = labels
        
        # Compute boundaries: where label changes
        diff = np.diff(labels)
        change_idx = np.where(diff != 0)[0] + 1  # +1 for end of previous
        starts = np.insert(change_idx, 0, 0)
        ends = np.append(change_idx, signal_len)
        activity_labels = labels[starts]
        boundaries[sub] = list(zip(starts, ends, activity_labels))
    
    return full_signals, sample_labels, boundaries

# Reconstruct train and test
train_signals, train_labels, train_boundaries = reconstruct_streams('train')
test_signals, test_labels, test_boundaries = reconstruct_streams('test')

# Combine for normalization
all_signals = np.vstack(list(train_signals.values()) + list(test_signals.values()))

# Normalize (Z-score per channel)
scaler = StandardScaler()
all_signals = scaler.fit_transform(all_signals)

# Reassign normalized signals back to dicts
offset = 0
for signals_dict in [train_signals, test_signals]:
    for sub in signals_dict:
        sig_len = len(signals_dict[sub])
        signals_dict[sub] = all_signals[offset:offset + sig_len]
        offset += sig_len

# Save for later use (e.g., in model training)
os.makedirs('processed_data', exist_ok=True)
np.save('processed_data/train_signals.npy', train_signals)
np.save('processed_data/train_sample_labels.npy', train_labels)
np.save('processed_data/train_boundaries.npy', train_boundaries)
np.save('processed_data/test_signals.npy', test_signals)
np.save('processed_data/test_sample_labels.npy', test_labels)
np.save('processed_data/test_boundaries.npy', test_boundaries)

print("Data preparation complete. Files saved in 'processed_data/'.")
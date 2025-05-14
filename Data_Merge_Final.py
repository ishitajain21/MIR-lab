import os
import json
import pandas as pd
import random
from collections import defaultdict
from itertools import chain

# === Configuration ===
FEATURES_ROOT = '/Users/arminhamrah/Downloads/MIR/asap_features'
OUTPUT_DIR = '/Users/arminhamrah/Downloads/MIR/seq2seq_datasets'
TRAIN_SPLIT = 0.5
TEST_SPLIT = 0.4  # Validation will implicitly be 0.1
RANDOM_SEED = 42
# ======================

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Gather all performance CSV files
piece_to_files = defaultdict(list)

for root, _, files in os.walk(FEATURES_ROOT):
    if 'midi_score.csv' in files:
        piece_name = os.path.relpath(root, FEATURES_ROOT)
        for f in files:
            if f.endswith('.csv') and f != 'midi_score.csv':
                piece_to_files[piece_name].append(os.path.join(root, f))

# Shuffle and split into train/test/val (at PIECE LEVEL)
all_pieces = list(piece_to_files.keys())
random.shuffle(all_pieces)

force_test_pieces = [
    'Bach/Prelude/bwv_880',
    'Brahms/Six_Pieces_op_118/2',
    'Chopin/Barcarolle',
    'Haydn/Keyboard_Sonatas/39-3',
    'Liszt/Ballade_2',
    'Mozart/Piano_Sonatas/12-1',
    'Prokofiev/Toccata',
    'Rachmaninoff/Preludes_op_32/10',
    'Schubert/Impromptu_op.90_D.899/2',
    'Schumann/Kreisleriana/3',
]

# # Convert to normalized absolute paths
# force_test_abs = [os.path.join(FEATURES_ROOT, p) for p in force_test_pieces]

# Filter out forced test pieces from the full list
normalized_forced = set(os.path.normpath(p) for p in force_test_pieces)
remaining_pieces = [p for p in all_pieces if os.path.normpath(p) not in normalized_forced]

# Shuffle and split the remaining pieces
n = len(remaining_pieces)
n_train = int(n * TRAIN_SPLIT)
n_test = int(n * TEST_SPLIT)

train_pieces = remaining_pieces[:n_train]
test_pieces = force_test_pieces + remaining_pieces[n_train:n_train + n_test]
val_pieces = remaining_pieces[n_train + n_test:]

# Flatten to file lists
train_files = list(chain.from_iterable(piece_to_files[p] for p in train_pieces if p in piece_to_files))
test_files  = list(chain.from_iterable(piece_to_files[p] for p in test_pieces if p in piece_to_files))
val_files   = list(chain.from_iterable(piece_to_files[p] for p in val_pieces if p in piece_to_files))

def write_list(file_list, list_name):
    list_path = os.path.join(OUTPUT_DIR, f'{list_name}.txt')
    with open(list_path, 'w') as f:
        for csv_path in file_list:
            rel_path = os.path.relpath(csv_path, FEATURES_ROOT)
            mid_path = os.path.splitext(rel_path)[0] + '.mid'
            full_path = os.path.join(FEATURES_ROOT, mid_path)
            f.write(full_path + '\n')
    print(f'Wrote {len(file_list)} entries to {list_path}')

write_list(train_files, 'train')
write_list(test_files, 'test')
write_list(val_files, 'val')

def merge_perf_pure(perf_csv):
    # Get corresponding pure MIDI CSV in same directory
    dirpath = os.path.dirname(perf_csv)
    pure_csv = os.path.join(dirpath, 'midi_score.csv')
    if not os.path.isfile(pure_csv):
        return pd.DataFrame()

    perf_df = pd.read_csv(perf_csv)
    pure_df = pd.read_csv(pure_csv)

    records = []
    measures = sorted(set(perf_df['measure']).intersection(pure_df['measure']))

    for m in measures:
        perf_sec = perf_df[perf_df['measure'] == m]
        pure_sec = pure_df[pure_df['measure'] == m]

        measure_start = perf_sec['measure_start'].iloc[0]
        measure_end = perf_sec['measure_end'].iloc[0]
        measure_dur = measure_end - measure_start

        # x = performance MIDI: (note, rel_onset, rel_duration, measure_dur)
        x_list = []
        for _, row in perf_sec.iterrows():
            rel_onset = row['onset'] - measure_start
            duration = row['offset'] - row['onset']
            x_list.append((row['note'], rel_onset, duration, measure_dur))

        # y = pure MIDI: (note, rel_onset, rel_offset)
        y_list = []
        for _, row in pure_sec.iterrows():
            rel_onset = row['onset'] - measure_start
            rel_offset = row['offset'] - measure_start
            y_list.append((row['note'], rel_onset, rel_offset))

        records.append({
            'perf_csv': perf_csv,
            'measure': m,
            'x': json.dumps(x_list),
            'y': json.dumps(y_list)
        })

    return pd.DataFrame(records)

# Build and save each dataset
for split_name, files in [('train', train_files), ('test', test_files), ('val', val_files)]:
    dfs = [merge_perf_pure(p) for p in files]
    df_split = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    out_path = os.path.join(OUTPUT_DIR, f'{split_name}.csv')
    df_split.to_csv(out_path, index=False)
    print(f'Wrote {len(df_split)} samples to {out_path}')

#1. Collects all performance CSVs under asap_features
#2. Shuffles and splits them into train/test/validation sets (80/10/10)
#3. Writes out train.list, test.list, and val.list listing the full .mid paths
#4. Merges each performance/pure CSV pair by measure, packing note‐level features (note, onset, offset, measure_duration, time_signature) into JSON lists (x, y)
#5. Saves train.csv, test.csv, and val.csv under seq2seq_datasets, one row per measure sample
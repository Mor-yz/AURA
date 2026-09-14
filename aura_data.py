"""AURA data contract: 12 separate motors, four signals, contiguous native frames."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

MOTOR_TYPES = ['HIP_PITCH', 'HIP_ROLL', 'HIP_YAW', 'KNEE_PITCH', 'ANKLE_PITCH', 'ANKLE_ROLL']
MOTOR_IDS = [f'{motor}_{side}_J{i + 1}' for side in ('L', 'R') for i, motor in enumerate(MOTOR_TYPES)]
FEATURES = ['cmd_effort', 'fb_effort', 'position', 'velocity']
TAU_MAX_VALUES = [360., 128., 128., 360., 101., 130.] * 2
WINDOW = 30


def read_metadata(csv_path):
    """Only explicit simulation labels are ground truth; folder names are not."""
    path = Path(csv_path).with_suffix('.metadata.json')
    metadata = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    labels = np.full((len(MOTOR_IDS), 2), np.nan, dtype=np.float32)
    if metadata.get('parameter_source') == 'simulation':
        for motor, values in metadata.get('parameters', {}).items():
            if motor not in MOTOR_IDS:
                raise ValueError(f'{path}: unknown motor {motor}')
            alpha, kappa = values
            if not (np.isfinite([alpha, kappa]).all() and 0 < alpha <= 1 and kappa > 0):
                raise ValueError(f'{path}: invalid parameters for {motor}')
            labels[MOTOR_IDS.index(motor)] = alpha, kappa
    return metadata, labels


def build_windows(csv_path, window=WINDOW, stride=15, max_gap_ms=50.0):
    """Return [N,12,H,4]; never join across missing frames, gaps, or recordings.

    stride selects window starts, not frames inside each window. Legacy CSVs
    retain their written timestamp precision; existing zero-filled commands
    cannot be diagnosed reliably and must be regenerated from the bags.
    """
    if window < 1 or stride < 1 or not np.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError('window, stride and max_gap_ms must be positive')
    df = pd.read_csv(csv_path, dtype={'timestamp': str})
    required = {'timestamp', 'motor_id', *FEATURES}
    if not required.issubset(df.columns):
        raise ValueError(f'{csv_path}: missing columns {sorted(required - set(df.columns))}')
    if 'timestamp_ns' not in df:
        df['timestamp_ns'] = df['timestamp'].map(lambda t: int(Decimal(t) * 1_000_000_000))
    key = 'frame_id' if 'frame_id' in df else 'timestamp_ns'
    windows, segment = [], []
    previous_ns = previous_frame = None

    def flush():
        for start in range(0, len(segment) - window + 1, stride):
            windows.append(np.stack(segment[start:start + window]).transpose(1, 0, 2))
        segment.clear()

    for frame_id, frame in df.groupby(key, sort=True):
        times = frame['timestamp_ns'].unique()
        if len(times) != 1:
            raise ValueError(f'{csv_path}: frame {frame_id} contains multiple timestamps')
        ns = int(times[0])
        if previous_ns is not None and (ns <= previous_ns or ns - previous_ns > max_gap_ms * 1e6
                                       or (key == 'frame_id' and frame_id != previous_frame + 1)):
            flush()
        previous_ns, previous_frame = ns, frame_id
        frame = frame[frame['motor_id'].isin(MOTOR_IDS)]
        if frame['motor_id'].duplicated().any():
            raise ValueError(f'{csv_path}: duplicate motor in frame {frame_id}')
        frame = frame.set_index('motor_id').reindex(MOTOR_IDS)
        values = frame[FEATURES].to_numpy(dtype=np.float32)
        valid = np.isfinite(values).all()
        if 'cmd_matched' in frame:
            valid = valid and frame['cmd_matched'].eq(1).all()
        if not valid:
            flush()
            continue
        segment.append(values)
    flush()
    return np.stack(windows) if windows else np.empty((0, 12, window, 4), dtype=np.float32)


def load_recordings(csv_dir, window=WINDOW, stride=15, max_gap_ms=50.0):
    records = []
    for path in sorted(Path(csv_dir).rglob('motor_data.csv')):
        metadata, labels = read_metadata(path)
        windows = build_windows(path, window, stride, max_gap_ms)
        if len(windows):
            records.append({'path': str(path.resolve()), 'session_id': metadata.get('session_id', str(path.resolve())),
                            'split': metadata.get('split'), 'windows': windows, 'labels': labels})
            print(f'{path.parent}: {len(windows)} windows')
    if not records:
        raise ValueError(f'No complete contiguous windows in {csv_dir}')
    return records


def split_recordings(records, seed=42):
    """Split entire sessions before normalization; explicit splits enable cross-motion tests."""
    groups = sorted({record['session_id'] for record in records})
    assigned = {}
    explicit = any(record['split'] is not None for record in records)
    if explicit:
        for record in records:
            split = record['split']
            if split not in ('train', 'val', 'test'):
                raise ValueError('With explicit splits, every recording must specify train, val or test')
            group = record['session_id']
            if group in assigned and assigned[group] != split:
                raise ValueError(f'Session {group} crosses dataset splits')
            assigned[group] = split
    else:
        if len(groups) < 3:
            raise ValueError('At least three independent sessions are required for train/val/test')
        shuffled = np.random.default_rng(seed).permutation(groups).tolist()
        n_holdout = max(1, len(groups) // 10)
        assigned = {group: ('test' if i < n_holdout else 'val' if i < 2 * n_holdout else 'train')
                    for i, group in enumerate(shuffled)}
    splits = {split: [r for r in records if assigned[r['session_id']] == split]
              for split in ('train', 'val', 'test')}
    if any(not values for values in splits.values()):
        raise ValueError('Train, validation and test must each contain usable recordings')
    return splits

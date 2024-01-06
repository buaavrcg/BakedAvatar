import numpy as np
import os
import json
import argparse


def read_flame_sequence(flame_params_path):
    with open(flame_params_path, 'r') as f:
        flame_params = json.load(f)
    exps = []
    poses = []
    for frame in flame_params['frames']:
        exps.append(np.array(frame["expression"], dtype=np.float32))
        poses.append(np.array(frame["pose"], dtype=np.float32))
    exps = np.stack(exps)
    poses = np.stack(poses)
    sequences = np.concatenate([exps, poses], axis=1)
    return sequences


def export_flame_sequences(flame_params_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total_length = 0
    sequences = []
    flame_sequences = {}
    for flame_params_path in flame_params_paths:
        seq = read_flame_sequence(flame_params_path)
        sequences.append(seq)
        total_length += len(seq)

        p = os.path.dirname(flame_params_path)
        pp = os.path.dirname(p)
        name = os.path.basename(pp) + '_' + os.path.basename(p)

        flame_sequences[name] = {
            'start': total_length - len(seq),
            'end': total_length,
        }
        print(f"Exported {name} with {len(seq)} frames.")

    sequences = np.concatenate(sequences, axis=0)
    np.save(os.path.join(output_dir, 'flame_sequences.npy'), sequences)
    with open(os.path.join(output_dir, 'flame_sequences.json'), 'w') as f:
        json.dump(flame_sequences, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('flame_params_paths', nargs='+', type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    args = parser.parse_args()

    export_flame_sequences(args.flame_params_paths, args.output_dir)
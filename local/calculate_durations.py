import argparse
import numpy as np
import os
import json
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--input-alignment-list', type=str)
parser.add_argument('--json', type=str)
parser.add_argument('--output-file', type=str)
parser.add_argument('--trans-type', type=str, default='phn')
parser.add_argument('--space', type=str, default='<space>')
parser.add_argument('--deconv-factor', type=int, default=4)
parser.add_argument('--offset', type=int, default=0)
args = parser.parse_args()

# read tokens and num_frames from .json file
with open(args.json, 'rb') as f:
    input_json = json.load(f)['utts']
utt2tokens = {}
utt2num_frames = {}
for uttid, info in input_json.items():
    tokens = [token.replace(args.space, 'sil') for token in info['output'][0]['token'].split()]
    num_frames = info['input'][0]['shape'][0]
    utt2tokens[uttid] = tokens
    utt2num_frames[uttid] = num_frames

# convert .npy alignment to kaldi ali file
with open(args.input_alignment_list, 'r') as fr, open(args.output_file, 'w') as fw:
    for fnpy in fr.readlines():
        # load info
        fnpy = fnpy.strip()
        uttid = os.path.basename(fnpy)[:-4]
        num_frames = utt2num_frames[uttid]
        tokens = utt2tokens[uttid]
        alignment = np.transpose(np.load(fnpy))
        T, U = alignment.shape
        assert len(tokens) == U - 1
        if not T > U:
            logging.warning(uttid + ' : T <= U')
            continue
        
        # dynamic programming
        weight_matrix = np.zeros_like(alignment)
        weight_matrix[:] = -np.inf
        weight_matrix[0, 0] = alignment[0, 0]
        direction_matrix = -np.ones_like(alignment, dtype=np.int32)
        for i in range(1,T):
            for j in range(min(i+1, U)):
                if j == 0:
                    weight_matrix[i, j], direction_matrix[i, j] = \
                        (weight_matrix[i-1, j] + alignment[i, j], 0)
                else:
                    weight_matrix[i, j], direction_matrix[i, j] = \
                        sorted([(weight_matrix[i-1, j-1] + alignment[i, j], 1),
                                (weight_matrix[i-1, j] + alignment[i, j], 0)])[-1]

        # trace back to find the best path
        path_matrix = np.zeros_like(alignment, dtype=np.int32)
        i = T -1
        j = U - 1
        path_matrix[i, j] = 1
        direction = 0
        while direction != -1:
            direction = direction_matrix[i, j]
            if direction == 0:
                i -= 1
            elif direction == 1:
                i -= 1
                j -= 1
            path_matrix[i, j] = 1
        assert i == 0 and j == 0

        # remove the first token, <sos>
        durations = np.sum(path_matrix, axis=0).tolist()
        durations[1] += durations[0]
        del durations[0]
        # offset and repeat
        durations = list(map(lambda x: x*args.deconv_factor, durations))
        durations[0] += args.offset
        durations[-1] += num_frames - sum(durations)
        # write out
        fw.write(uttid + ' ' + ' ; '.join([t + ' ' + str(d) for t, d in zip(tokens, durations)]) + '\n')

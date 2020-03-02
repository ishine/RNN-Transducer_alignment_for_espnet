import argparse
import numpy as np
import os
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-alignment-list', type=str)
parser.add_argument('--json', type=str)
parser.add_argument('--output-file', type=str)
parser.add_argument('--trans-type', type=str, default='phn')
parser.add_argument('--space', type=str, default='<space>')
parser.add_argument('--deconv-factor', type=int, default=4)
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
    for fnpy in tqdm(fr.readlines()):
        # load info
        fnpy = fnpy.strip()
        uttid = os.path.basename(fnpy)[:-4]
        num_frames = utt2num_frames[uttid]
        tokens = utt2tokens[uttid]
        alignment = np.transpose(np.load(fnpy))
        T, U = alignment.shape
        assert len(tokens) == U - 1

        u_idx = alignment.argmax(axis=-1)
        smoothing_frame = 0
        durations = []
        for u in range(U):
            duration = (u_idx == u).sum() * args.deconv_factor
            if u == U - 1:
                duration += num_frames - T * args.deconv_factor

            # smoothing
            if duration == 0:
                duration = 1
                smoothing_frame += 1
                if smoothing_frame > 2:
                    break
            elif smoothing_frame > 0:
                x = min(smoothing_frame, duration - 1)
                duration -= x
                smoothing_frame -= x

            durations.append(duration)

        if not smoothing_frame > 0:
            # convert length of 'durations' frome U - 1 to U.
            durations[1] += durations[0]
            del durations[0]
            # write out
            fw.write(uttid + ' ' + ' ; '.join([t + ' ' + str(d) for t, d in zip(tokens, durations)]) + '\n')




import numpy as np
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input-alignment-list', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--g', type=float, default=0.2, help='A hyper parameter.')
args = parser.parse_args()


utt2cost = {}
with open(args.input_alignment_list, 'r') as f:
    for fnpy in tqdm(f.readlines()):
        fnpy = fnpy.strip()
        alignment = np.transpose(np.load(fnpy))
        T, U = alignment.shape
        W = 1 - np.exp(-(np.expand_dims(np.arange(T), axis=1) / T - np.expand_dims(np.arange(U), axis=0) / U) ** 2 / (2 * args.g ** 2))
        cost = W[np.arange(T), np.argmax(alignment, axis=1)].mean()
        utt2cost[fnpy] = cost

import pickle
with open(os.path.join(args.outdir, 'utt2cost.dict'), 'wb') as f:
    pickle.dump(utt2cost, f)

import matplotlib.pyplot as plt
fig = plt.hist(utt2cost.values(), bins=500)
plt.savefig(os.path.join(args.outdir, 'hist.png'))


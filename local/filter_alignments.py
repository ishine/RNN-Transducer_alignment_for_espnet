import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--utt2cost', type=str)
parser.add_argument('--thres', type=float)
parser.add_argument('--outdir', type=str)
args = parser.parse_args()

with open(args.utt2cost, 'rb') as f:
    utt2cost = pickle.load(f)

with open(os.path.join(args.outdir, 'out.list'), 'w') as f:
    for utt, cost in utt2cost.items():
        if cost < args.thres:
            f.write(utt+'\n')

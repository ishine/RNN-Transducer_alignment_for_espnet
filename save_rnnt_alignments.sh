#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

stage=0
stop_stage=100
. utils/parse_options.sh || exit 1;

# general configuration
backend=pytorch
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
batchsize=24

expdir=exp/train_nodup_pytorch_train_transducer
model=${expdir}/results/snapshot.ep.xx
json=dump/train_nodup/deltafalse/data.json

npydir=${expdir}/calculate_alignments/`basename ${model}`/`basename ${json} .json`
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Calculate RNN-T alignments."
    mkdir -p ${npydir}
    python asr_custom.py \
        --custom-task save_alignment \
        --batchsize ${batchsize} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --model ${model} \
        --json ${json} \
        --outdir ${npydir}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Select diagnal alignments."
    outdir=${expdir}/filt_alignments/`basename ${model}`/`basename ${json} .json`
    mkdir -p ${outdir}
    find ${npydir} -iname "*.npy" > ${outdir}/in.list
    python local/calculate_cost.py \
        --input-alignment-list  ${outdir}/in.list \
        --outdir ${outdir} \
        --g 0.2
    python local/filter_alignments.py \
        --utt2cost ${outdir}/utt2cost.dict \
        --thres 0.22 \
        --outdir ${outdir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Write alignments in kaldi format."
    python local/calculate_durations.py \
         --input-alignment-list ${outdir}/out.list \
         --json ${json} \
         --output-file ${outdir}/ali.txt
fi

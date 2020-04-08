#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

stage=0
stop_stage=100
nj=20
. utils/parse_options.sh || exit 1;

# general configuration
backend=pytorch
batchsize=24

expdir=exp/train_nodup_pytorch_train
model=${expdir}/results/snapshot.ep.100
json=dump/train_nodup/deltafalse/data.json

npydir=${expdir}/calculate_alignments/`basename ${model}`/`basename ${json} .json`
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Calculate alignments."
    mkdir -p ${npydir}
    python asr_custom.py \
        --custom-task save_alignment \
        --batchsize ${batchsize} \
        --ngpu 1 \
        --backend ${backend} \
        --model ${model} \
        --json ${json} \
        --outdir ${npydir}
fi

outdir=${expdir}/filt_alignments/`basename ${model}`/`basename ${json} .json`
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Select diagnal alignments."
    mkdir -p ${outdir}
    find ${npydir} -iname "*.npy" > ${outdir}/in.list
    python local/calculate_cost.py \
        --input-alignment-list ${outdir}/in.list \
        --outdir ${outdir} \
        --g 0.2
    python local/filter_alignments.py \
        --utt2cost ${outdir}/utt2cost.dict \
        --thres 0.22 \
        --outdir ${outdir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Write alignments in kaldi format."

    mkdir -p ${outdir}/split${nj}
    for n in $(seq $nj); do
        part=$(($n-1))
        ./utils/split_scp.pl -j ${nj} ${part} ${outdir}/out.list ${outdir}/split${nj}/out.${n}.list
    done

    ${train_cmd} JOB=1:${nj} ${outdir}/log/JOB.log \
        python local/calculate_durations.py \
             --input-alignment-list ${outdir}/split${nj}/out.JOB.list \
             --json ${json} \
             --offset 2 \
             --deconv-factor 4 \
             --output-file ${outdir}/log/ali.JOB.txt
        
    cat ${outdir}/log/ali.*.txt > ${outdir}/ali.txt
fi

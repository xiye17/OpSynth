#!/bin/bash
set -e

model_name0=$(basename $1)
split=$2
order=${3:-dfs}

test_file="data/streg/${split}.bin"
max_decode_step=70
beam_size=10
smooth='none'
smooth_alpha=0
model_name="${model_name0}.upbound"
decode_file="outputs/synth.decode.${model_name}.${split}.${order}.txt"
report_file="outputs/synth.report.${model_name}.${split}.${order}.txt"
eval_file="outputs/synth.eval.${model_name}.${split}.${order}.pkl"
cache_file="misc/cache.pkl"

mkdir -p logs

python -u synthesize.py \
    --test_file ${test_file} \
    --model_file $1 \
    --beam_size ${beam_size} \
    --search_order ${order} \
    --report_file ${report_file} \
    --decode_file ${decode_file} \
    --cache_file ${cache_file} \
    --eval_file ${eval_file} \
    --smooth ${smooth} \
    --smooth_alpha ${smooth_alpha} \
    --max_decode_step 70 2>&1 | tee -a logs/test.${model_name}.${split}.log

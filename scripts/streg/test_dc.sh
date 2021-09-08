#!/bin/bash
set -e

model_name=$(basename $1)
split=$2

test_file="data/streg/${split}.bin"
max_decode_step=70
beam_size=20
decode_file="outputs/dc.priors.${model_name}.${split}.txt"
report_file="outputs/synth.report.${model_name}.${split}.${order}.txt"
eval_file="outputs/synth.eval.${model_name}.${split}.${order}.pkl"
cache_file="misc/cache.pkl"

python -u test_dc.py \
    --test_file ${test_file} \
    --model_file $1 \
    --beam_size ${beam_size} \
    --report_file ${report_file} \
    --decode_file ${decode_file} \
    --cache_file ${cache_file} \
    --eval_file ${eval_file} \
    --max_decode_step 70 2>&1 | tee -a logs/test.${model_name}.${split}.log

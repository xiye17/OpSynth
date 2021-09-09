#!/bin/bash
set -e

model_name=$(basename $1)
split=$2
order=${3:-dfs}

test_file="data/streg/${split}.bin"
max_decode_step=70
beam_size=20
decode_file="outputs/streg.decode.${model_name}.b${beam_size}.${split}.${order}.bin"
report_file="outputs/streg.report.${model_name}.${split}.${order}.txt"
eval_file="outputs/streg.eval.${model_name}.${split}.${order}.pkl"
cache_file="misc/cache.pkl"

mkdir -p logs

python -u test.py \
    --test_file ${test_file} \
    --model_file $1 \
    --beam_size ${beam_size} \
    --search_order ${order} \
    --report_file ${report_file} \
    --decode_file ${decode_file} \
    --cache_file ${cache_file} \
    --eval_file ${eval_file} \
    --max_decode_step 70 2>&1 | tee -a logs/test.${model_name}.${split}.log

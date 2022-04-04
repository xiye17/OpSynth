#!/bin/bash
set -e

seed=${1:-0}
asdl_file="data/streg/streg_asdl.txt"
vocab="data/streg/vocab.bin"
io_vocab="data/streg/io_vocab.bin"
train_file="data/streg/train.bin"
dev_file="data/streg/dev.bin"
dropout=0.3
io_hid_size=100
io_emb_size=50
enc_hid_size=100
field_emb_size=100
max_epoch=200
clip_grad=5.0
batch_size=32
lr=0.003
ls=0.1
model_file=fill.streg.ioenc${enc_hidden_size}.ioemb${src_emb_size}.enc${enc_hid_size}.field${field_emb_size}.drop${dropout}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.clip_grad${clip_grad}.bin

# echo "**** Writing results to logs/regex/${model_name}.log ****"
# mkdir -p logs/regex
# echo commit hash: `git rev-parse HEAD` > logs/regex/${model_name}.log
mkdir -p logs

python -u train_fill.py \
    --asdl_file ${asdl_file} \
    --train_file ${train_file} \
    --model_name RobustFill \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --io_vocab ${io_vocab} \
    --io_hid_size ${io_hid_size} \
    --io_emb_size ${io_emb_size} \
    --enc_hid_size ${enc_hid_size} \
    --field_emb_size ${field_emb_size} \
    --dropout ${dropout} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --clip_grad ${clip_grad} \
    --log_every 50 \
    --run_val_after 20 \
    --max_decode_step 70 \
    --save_all \
    --save_to checkpoints/streg/${model_file} 2>&1 | tee -a logs/${model_file}.log

# . scripts/streg/test_fill.sh checkpoints/streg/${model_file} testi
# . scripts/streg/test_fill.sh checkpoints/streg/${model_file} teste
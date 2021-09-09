#!/bin/bash
set -e

seed=${1:-0}
asdl_file="data/streg/streg_asdl.txt"
vocab="data/streg/vocab.bin"
train_file="data/streg/train.bin"
dev_file="data/streg/dev.bin"
dropout=0.3
enc_hid_size=100
src_emb_size=100
field_emb_size=100
# type_embed_size=32
# lr_decay=0.985
# lr_decay_after_epoch=20
max_epoch=200
# patience=5 
# beam_size=5
clip_grad=5.0
batch_size=32
lr=0.003
ls=0.1
# lstm='lstm'
model_file=model.streg.enc${enc_hidden_size}.src${src_emb_size}.field${field_emb_size}.drop${dropout}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.clip_grad${clip_grad}.bin

# echo "**** Writing results to logs/regex/${model_name}.log ****"
# mkdir -p logs/regex
# echo commit hash: `git rev-parse HEAD` > logs/regex/${model_name}.log
mkdir -p logs

python -u train.py \
    --asdl_file ${asdl_file} \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --enc_hid_size ${enc_hid_size} \
    --src_emb_size ${src_emb_size} \
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

. scripts/streg/test.sh checkpoints/streg/${model_file} testi
. scripts/streg/test.sh checkpoints/streg/${model_file} teste
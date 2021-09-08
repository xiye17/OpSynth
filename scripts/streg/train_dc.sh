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
io_pooling='max'
field_emb_size=100
max_epoch=200
clip_grad=5.0
batch_size=32
lr=0.003
model_file=dc.streg.ioenc${enc_hidden_size}.ioemb${src_emb_size}.pool${io_pooling}.drop${dropout}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.clip_grad${clip_grad}.bin

python -u train_dc.py \
    --asdl_file ${asdl_file} \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --io_vocab ${io_vocab} \
    --io_hid_size ${io_hid_size} \
    --io_emb_size ${io_emb_size} \
    --io_pooling ${io_pooling} \
    --dropout ${dropout} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --batch_size ${batch_size} \
    --clip_grad ${clip_grad} \
    --log_every 50 \
    --run_val_after 10 \
    --save_to checkpoints/streg/${model_file} 2>&1 | tee -a logs/${model_file}.log

# . scripts/streg/test_dc.sh checkpoints/streg/${model_file} testi
# . scripts/streg/test_dc.sh checkpoints/streg/${model_file} teste
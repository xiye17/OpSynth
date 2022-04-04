# OpSynth


Code for the paper [Optimal Neural Program Synthesis from Multimodal Specifications](https://arxiv.org/abs/2010.01678) (Findings of EMNLP, 2021).

```bibtex
@inproceedings{ye2021optimal,
  title={Optimal Neural Program Synthesis from Multimodal Specifications},
  author = {Xi Ye, Qiaochu Chen, Isil Dillig, and Greg Durrett},
  booktitle = {Findings of EMNLP},
  year={2021}
}

```


## Requirements
* python==3.8
* pytorch==1.6.0
* JAVA 1.8.0

## Code
We've already attached trained checkpoint at `checkpoints/streg/streg.enc.src100.field100.bin`.

**Preprocess data**

```
python -c 'from datasets.streg.make_dataset import make_dataset;make_dataset()'
```

**Run Optimal Synthesis**

```
# <split>: the split (dev,testi, or teste) to evaluate on.
sh scripts/streg/synth.sh checkpoints/streg/streg.src100.field100.bin <split>
```

**Train a Model**

If you'd like to train a new ASN model, run the following command. The checkpoints will be stored at `checkpoints/streg/`
```
sh scripts/streg/train.sh
```

**Run RobustFill**

```
python -c 'from datasets.streg.make_deepcoder_data import make_exs_vocab;make_exs_vocab()'

sh scripts/streg/test_fill.sh checkpoints/streg/streg.robustfill.ioenc100.src100.field100.bin teste
```

**Train RobustFill**

```
sh scripts/streg/train_fill.sh
```

## Credit
Part of the codes and system design are modified from [TranX](https://github.com/pcyin/tranX).
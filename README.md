# Differential-Privacy-Tiny-Bert
Training and evaluation of tiny-bert model on SST2, QNLI, QQP, and MNLI with differential privacy

## Models

This repository uses Tiny-BERT, available at https://huggingface.co/prajjwal1/bert-tiny, with eight different fine-tuning techniques:
 - soft-prompt
 - prefix
 - LoRA
 - full-finetuning
 - last-layer-finetuning
 - soft-prompt + LoRA
 - prefix + LoRA
 - (IA)^3


## Preliminary

To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate dp
```
### Evaluated tasks

The evaluation datasets are SST2, QNLI, QQP, and MNLI.

## Training

For different fine-tuning techniques, you can run the corresponding `.ipynb` file. The number of epochs is set to 5, the learning rate is 1e-2, and the privacy budget (ε) is 8.

## Result 
The results of all the fine-tuning techniques can be accessed at https://docs.google.com/spreadsheets/d/1EMXQVGBWbaSlTqnagKM_5kQbhThmp1qfOrLUMgfb7TA/edit?usp=sharing

#!/bin/bash
#learning_rate=0.1
batch_size=2

i=1
for learning_rate in 0.02 0.1 0.5
do
  for rep in 1 2 3 4 5
  do
    python train.py --dataset 1dfunction --network fnn --n_samples_per_class 10 --n_iters 100000 --batch_size $batch_size --load_size $batch_size --learning_rate $learning_rate --model_file 1d/model_lr${i}_${rep}.pkl
  done
  i=$((i+1))
done





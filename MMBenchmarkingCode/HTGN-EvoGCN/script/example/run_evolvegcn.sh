#!/usr/bin/env bash
device_id=0
for dataset in as733 dblp enron10 fbw
do
  for model in EGCN
  do
    python ./baselines/run_evolvegcn_baselines.py \
          --model=${model} \
          --dataset=${dataset} \
          --device_id=${device_id}
  done
done
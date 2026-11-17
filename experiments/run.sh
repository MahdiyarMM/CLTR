#!/usr/bin/env bash
# All reported runs (seeds 0-2). Runs one job per GPU listed in GPUS.
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
jobs=()
for s in 0 1 2; do
  for c in c100_if100 c100_if50 c100_if10 c10_if100 c10_if50; do jobs+=("cifar configs/cifar/$c.yaml $s"); done
  for c in if100 if50 if10; do jobs+=("clip configs/clip/$c.yaml $s"); done
done
i=0
for j in "${jobs[@]}"; do
  set -- $j
  CUDA_VISIBLE_DEVICES=${GPUS[$((i % ${#GPUS[@]}))]} python main.py $1 --config $2 --seed $3 &
  i=$((i + 1))
  (( i % ${#GPUS[@]} == 0 )) && wait
done
wait

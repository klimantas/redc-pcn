#!/bin/bash

python -m tools.train_eval \
  --start_seed 0 \
  --stop_seed 9 \
  --device 0 \
  --dataset SR-GRAPHS \
  --exp_name gin-sr \
  --model gin \
  --drop_rate 0.0 \
  --nonlinearity elu \
  --readout sum \
  --lr_scheduler None \
  --num_layers 6 \
  --emb_dim 16 \
  --batch_size 8 \
  --num_workers 8 \
  --task_type isomorphism \
  --eval_metric isomorphism \
  --note "gin-sr baseline" \
  --untrained

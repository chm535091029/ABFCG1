#!/bin/bash

python main.py \
  --train_path "../../processed datasets/DEEPCOM/deepcom-train_samples_ids.json" \
  --valid_path "../../processed datasets/DEEPCOM/deepcom-valid_samples_ids.json" \
  --test_path "../../processed datasets/DEEPCOM/deepcom-test_samples_ids.json" \
  --batch_size_train 12 \
  --batch_size_eval 16 \
  --num_epochs 40 \
  --dataset_name "deepcom" \
  --load_weight "v1_last_deepcom.pt" \
  --save_weight "v2.pt" \
  --test_weight "v2_test.pt" \
  --early_stop \
  --patience 8

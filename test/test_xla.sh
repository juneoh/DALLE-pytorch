#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TEMP_DIR=$(mktemp -d)

function run_xla {
  "$@" --fake_data --distributed_backend=XLA --tpu_cores=8 \
    --batch_size=8 --epochs=2 --wandb_mode=disabled
}

run_xla python $CDIR/../train_vae.py --model_dir=$TEMP_DIR
run_xla python $CDIR/../train_dalle.py --dalle_output_file_name=$TEMP_DIR/dalle

rm -rf $TEMP_DIR
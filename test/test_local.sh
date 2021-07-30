#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TEMP_DIR=$(mktemp -d)

function run_local {
    "$@" --fake_data --epochs=2 --wandb_mode=disabled
}

run_local python $CDIR/../train_vae.py --model_dir=$TEMP_DIR
run_local python $CDIR/../train_dalle.py --dalle_output_file_name=$TEMP_DIR/dalle

rm -rf $TEMP_DIR
#!/bin/bash

# Directory where you want to clone the repo
DEST_DIR="./models"

cd $DEST_DIR

curl -L https://huggingface.co/state-spaces/mamba-2.8b-slimpj/raw/refs%2Fpr%2F1/config.json -o ./config.json
curl -L https://huggingface.co/EleutherAI/gpt-neox-20b/raw/main/tokenizer.json -o ./tokenizer.json
curl -L https://huggingface.co/state-spaces/mamba-2.8b-slimpj/resolve/refs%2Fpr%2F1/model.safetensors -o ./model.safetensors

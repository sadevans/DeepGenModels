#!/bin/bash
mkdir -p cache_dir
cd cache_dir
mkdir -p models
cd ..
cd ..

# Download config file from GitHub
wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml

# Download model from CivitAI
wget -O civitai_model.ckpt https://civitai.com/api/download/models/201259?type=Model&format=SafeTensor&size=pruned&fp=fp16

echo "Downloads completed!"

python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py \
      --checkpoint_path='./civitai_model.ckpt'\
      --original_config_file='./v1-inference.yaml'\
      --dump_path='./cache_dir/models/civitai_model'\
      --scheduler_type="ddim" --prediction_type='epsilon'\
      --from_safetensors
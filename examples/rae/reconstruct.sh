# python tools/merge_fsdp.py --input_dir=/mnt/bn/seed-aws-va/brianli/output/rae_siglip_v2 --type=fsdp2

# Use --use_ema flag to load EMA weights for better reconstruction quality
python examples/rae/reconstruct.py \
    --model_path=./output/rae_siglip_v2 \
    --output_path=./output/rae_siglip_v2/reconstructed_ema.png \
    --use_ema
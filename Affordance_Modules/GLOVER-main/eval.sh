CUDA_VISIBLE_DEVICES=0 python eval.py \
    --dataset_dir='/path/to/HOVA-500K/datasets' \
    --version="/path/to/GLOVER(++) model" \
    --vision-tower="/path/to/clip-vit-large-patch14" \
    --model_arch="glover++" \
    --use_text_emb_in_suffix_sam

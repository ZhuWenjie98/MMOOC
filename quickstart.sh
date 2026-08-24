CUDA_VISIBLE_DEVICES=1 python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mmstar \
  --batch_size 1 \
  --limit 10
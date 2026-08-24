# export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes=1 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,max_pixels=3240000 \
    --tasks=mme \
    --batch_size=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/include/x86_64-linux-gnu/
export PATH=$PATH:$HOME/.local/bin
export PROJECT_ROOT=$HOME/coedit
deepspeed --num_gpus=4 transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path google/flan-t5-xl \
    --output_dir $PROJECT_ROOT/output_coedit_xl \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --do_train \
    --do_eval \
    --train_file $PROJECT_ROOT/data/train.json \
    --validation_file $PROJECT_ROOT/data/val.json \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --source_lang "src" \
    --target_lang "tgt" \
    --deepspeed ds_config.json

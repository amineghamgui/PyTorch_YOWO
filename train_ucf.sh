# Train YOWO-D19
python train.py \
        --cuda \
        -d ucf24 \
        -v yowo_v2_ \
        --num_workers 4 \
        --eval_epoch 1 \
        --eval \
        # --fp16 \

#!/bin/bash

# TODO - run your inference Python3 code
# Find at which epochs model pass the baseline (david_revoy)

for i in $(seq 0 10 490)
do
    python3 /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/P3_inference.py  \
        --trained_embeddings_dog "../checkpoint_model/P3/trained_embedding/trained_embeddings_dog_epoch0.pt" \
        --trained_embeddings_david_revoy "../checkpoint_model/P3/trained_embedding/trained_embeddings_david_revoy_epoch$i.pt"
    python3 evaluation/grade_hw2_3.py \
        --json_path /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/hw2_data/textual_inversion/input.json \
        --input_dir /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/hw2_data/textual_inversion \
        --output_dir /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/P3_temp_out_3 >> P3_log_fine_tune_2/output$i.log 2>&1 
done

echo "finished."
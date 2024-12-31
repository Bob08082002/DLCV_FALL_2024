#!/bin/bash

# TODO - run your inference Python3 code
# Find at which epochs model pass the strong line


#for i in $(seq 0 1 4)
for i in $(seq 3)
do  
    python3 P2_inference.py /home/zhanggenqi/DLCV/HW3/dlcv-fall-2024-hw3-Bob08082002/hw3_data/p2_data/images/val \
            /home/zhanggenqi/DLCV/HW3/dlcv-fall-2024-hw3-Bob08082002/P2_temp_out/captions_output_v71_finetune2_ep$i.json \
            /home/zhanggenqi/DLCV/HW3/dlcv-fall-2024-hw3-Bob08082002/hw3_data/p2_data/decoder_model.bin \
            ./checkpoint_model/P2/P2_model_v7_1_finetune2_epoch$i.pth
    python3 ./evaluation/evaluate.py \
    --pred_file /home/zhanggenqi/DLCV/HW3/dlcv-fall-2024-hw3-Bob08082002/P2_temp_out/captions_output_v71_finetune2_ep$i.json \
    --images_root /home/zhanggenqi/DLCV/HW3/dlcv-fall-2024-hw3-Bob08082002/hw3_data/p2_data/images/val  \
    --annotation_file /home/zhanggenqi/DLCV/HW3/dlcv-fall-2024-hw3-Bob08082002/hw3_data/p2_data/val.json >> P2_v71_inference_log/finetune2_output$i.log 2>&1 
done

echo "finished."
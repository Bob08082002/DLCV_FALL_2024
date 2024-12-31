#!/bin/bash

# TODO - run your inference Python3 code


for i in {0..3}
do
    python3 ../P2_inference.py  /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/hw2_data/face/noise /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/P2_temp_output /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/hw2_data/face/UNet.pt $i
    python3 P2_evaluate_MSE.py /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/P2_temp_output /home/zhanggenqi/DLCV/HW2/dlcv-fall-2024-hw2-Bob08082002/hw2_data/face/GT
    
done

echo "finished."
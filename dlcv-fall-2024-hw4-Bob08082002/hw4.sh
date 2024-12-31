#!/bin/bash
python3 ./gaussian-splatting/MyRender.py -m ./checkpoint_model/model_v16 -s $1 --output_folder $2
# TODO - run your inference Python3 code
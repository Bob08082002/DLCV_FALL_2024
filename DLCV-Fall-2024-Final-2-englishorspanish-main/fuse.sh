# fuse all concepts
config_file="merge_all_chilloutmix"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50


# fuse cat2 and dog6
config_file="cat2_dog6_chilloutmix"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50

# fuse flower_1 and vase 
config_file="flower_1_vase_chilloutmix"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50

# fuse dog and pet_cat1 and dog6
config_file="dog_pet_cat1_dog6_chilloutmix"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50

# fuse cat2 wearable_glasses watercolor character
config_file="cat2_wearable_glasses_watercolor_chilloutmix"

python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${config_file}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50
#---------------------------------------------cat_2 and dog_6-------------------------------------------
# finish mmmb modify
real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix_0_7/cat2_dog6_chilloutmix/combined_model_base"

  sketch_adaptor_weight=1.0

  context_neg_prompt='cropped, worst quality, low quality'

  token1='[<cat21> <cat22>]'
  region1_prompt='[a <cat21> <cat22>, sitting on the right, high quality, high resolution, best quality, whole body]'
  region1_neg_prompt="[${context_neg_prompt}]"

  token2='[<dog61> <dog62>]'
  region2_prompt='[a <dog61> <dog62>, sitting on the left, high resolution, best quality, whole body]'
  region2_neg_prompt="[${context_neg_prompt}]"

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}|${region2_prompt}-*-${region2_neg_prompt}"
  token_rewrite="${token1}|${token2}"

  python3 mask_pipeline_mmmb.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --seed=31 \
    --image_height=512 \
    --image_width=512 \
    --token="${token_rewrite}" \
    --mask_dir="mmmb/prompt_0/merge_mask" \
    --save_dir_single_concept="mmmb/prompt_0" \
    --mask_num=5 \
    --gui_bbox_num=8 \
    --merge_mask_num=100

fi

#---------------------------------------------flower_1 and vase-------------------------------------------

# real_character=1

# if [ ${real_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/chilloutmix/flower_1_vase_chilloutmix/combined_model_base"
#   expdir="flower_1_vase_chilloutmix"

#   keypose_condition=''
#   keypose_adaptor_weight=1.0

#   sketch_condition='final_dataset/merge_mask/flower_1_vase/flower_1_vase_mask.png'
#   sketch_adaptor_weight=1.0

#   context_prompt='A flower in a vase.'
#   context_neg_prompt='cropped, worst quality, low quality'

#   token1='[<flower_11> <flower_12>]'
#   region1_prompt='[a <flower_11> <flower_12>, in a vase, 4K, high quality, high resolution, best quality]'
#   region1_neg_prompt="[${context_neg_prompt}]"

#   token2='[<vase1> <vase2>]'
#   region2_prompt='[a <vase1> <vase2>, 4K, high quality, high resolution, best quality]'
#   region2_neg_prompt="[${context_neg_prompt}]"

#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}|${region2_prompt}-*-${region2_neg_prompt}"
#   token_rewrite="${token1}|${token2}"

#   python3 mask_pipeline2.py \
#     --pretrained_model=${fused_model} \
#     --sketch_adaptor_weight=${sketch_adaptor_weight}\
#     --sketch_condition=${sketch_condition} \
#     --keypose_adaptor_weight=${keypose_adaptor_weight}\
#     --keypose_condition=${keypose_condition} \
#     --save_dir="results/multi-concept/${expdir}" \
#     --prompt="${context_prompt}" \
#     --negative_prompt="${context_neg_prompt}" \
#     --prompt_rewrite="${prompt_rewrite}" \
#     --suffix="baseline" \
#     --seed=7 \
#     --num_image=10 \
#     --image_height=1024 \
#     --image_width=512 \
#     --token="${token_rewrite}" \
#     --mask_dir="final_dataset/merge_mask/flower_1_vase"
# fi

#---------------------------------------------dog_pet_cat1_dog6------------------------------------------- 
# finish mmmb modify

# real_character=1

# if [ ${real_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/chilloutmix/dog_pet_cat1_dog6_chilloutmix/combined_model_base"

#   sketch_adaptor_weight=1.0

#   context_neg_prompt='cropped, worst quality, low quality, blur'

#   token1='[<dog1> <dog2>]'
#   region1_prompt='[a <dog1> <dog2>, dog, 4K, high quality, high resolution, best quality, whole body]'
#   region1_neg_prompt="[${context_neg_prompt}]"


#   token2='[<pet_cat11> <pet_cat12>]'
#   region2_prompt='[a <pet_cat11> <pet_cat12>, cat, 4K, high quality, high resolution, best quality, whole body]'
#   region2_neg_prompt="[${context_neg_prompt}]"


#   token3='[<dog61> <dog62>]'
#   region3_prompt='[a <dog61> <dog62>, dog, 4K, high quality, high resolution, best quality, whole body]'
#   region3_neg_prompt="[${context_neg_prompt}]"


#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}|${region2_prompt}-*-${region2_neg_prompt}|${region3_prompt}-*-${region3_neg_prompt}"
#   token_rewrite="${token1}|${token2}|${token3}"

#   python3 mask_pipeline_mmmb.py \
#     --pretrained_model=${fused_model} \
#     --sketch_adaptor_weight=${sketch_adaptor_weight}\
#     --sketch_condition=${sketch_condition} \
#     --negative_prompt="${context_neg_prompt}" \
#     --prompt_rewrite="${prompt_rewrite}" \
#     --seed=2 \
#     --image_height=512 \
#     --image_width=1024 \
#     --token="${token_rewrite}" \
#     --mask_dir="mmmb/prompt_2/merge_mask" \
#     --save_dir_single_concept="mmmb/prompt_2" \
#     --mask_num=5 \
#     --gui_bbox_num=8 \
#     --merge_mask_num=100


# fi

#---------------------------------------------cat2 wearable_glasses watercolor-------------------------------------------

# real_character=1

# if [ ${real_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/chilloutmix/cat2_wearable_glasses_watercolor_chilloutmix/combined_model_base"
#   expdir="cat2_wearable_glasses_watercolor_chilloutmix"

#   keypose_condition=''
#   keypose_adaptor_weight=1.0

#   sketch_condition='final_dataset/merge_mask/cat2_wearable_glasses_watercolor/cat2_wearable_glasses_watercolor_mask.png'
#   sketch_adaptor_weight=1.0

#   context_prompt='A cat wearing wearable glasses in a <watercolor1> <watercolor2> style.'
#   context_neg_prompt='cropped, worst quality, low quality'

#   token1='[<cat21> <cat22>]'
#   region1_prompt='[a <cat21> <cat22>, wearing wearable glasses, <watercolor1> <watercolor2> style, high quality, high resolution, best quality]'
#   region1_neg_prompt="[${context_neg_prompt}]"

#   token2='[<wearable> <glasses>]'
#   region2_prompt='[a cat with <wearable> <glasses>, high quality, <watercolor1> <watercolor2> style, high resolution, best quality]'
#   region2_neg_prompt="[${context_neg_prompt}]"

#   region3_prompt='[<watercolor1> <watercolor2>]'
#   region3_neg_prompt="[${context_neg_prompt}]"



#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}|${region2_prompt}-*-${region2_neg_prompt}|${region3_prompt}-*-${region3_neg_prompt}"
#   token_rewrite="${token1}|${token2}"

#   # prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

#   python3 mask_pipeline2.py \
#     --pretrained_model=${fused_model} \
#     --sketch_adaptor_weight=${sketch_adaptor_weight}\
#     --sketch_condition=${sketch_condition} \
#     --keypose_adaptor_weight=${keypose_adaptor_weight}\
#     --keypose_condition=${keypose_condition} \
#     --save_dir="results/multi-concept/${expdir}" \
#     --prompt="${context_prompt}" \
#     --negative_prompt="${context_neg_prompt}" \
#     --prompt_rewrite="${prompt_rewrite}" \
#     --suffix="baseline" \
#     --seed=7 \
#     --num_image=10 \
#     --image_height=512 \
#     --image_width=512 \
#     --token="${token_rewrite}" \
#     --mask_dir="final_dataset/merge_mask/cat2_wearable_glasses_watercolor"
# fi

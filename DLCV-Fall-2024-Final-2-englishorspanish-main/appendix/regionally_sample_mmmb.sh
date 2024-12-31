#---------------------------------------------anime-------------------------------------------

# anime_character=0

# if [ ${anime_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/anythingv4/hina+kario+tezuka_anythingv4/combined_model_base"
#   expdir="hina+kario+tezuka_anythingv4"

#   keypose_condition='datasets/validation_spatial_condition/multi-characters/anime_pose_2x/hina_tezuka_kario_2x.png'
#   keypose_adaptor_weight=1.0
#   sketch_condition=''
#   sketch_adaptor_weight=1.0

#   context_prompt='two girls and a boy are standing near a forest'
#   context_neg_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

#   region1_prompt='[a <hina1> <hina2>, standing near a forest]'
#   region1_neg_prompt="[${context_neg_prompt}]"
#   region1='[12, 36, 1024, 600]'

#   region2_prompt='[a <tezuka1> <tezuka2>, standing near a forest]'
#   region2_neg_prompt="[${context_neg_prompt}]"
#   region2='[18, 696, 1024, 1180]'

#   region5_prompt='[a <kaori1> <kaori2>, standing near a forest]'
#   region5_neg_prompt="[${context_neg_prompt}]"
#   region5='[142, 1259, 1024, 1956]'

#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region5_prompt}-*-${region5_neg_prompt}-*-${region5}"

#   python3 regionally_controlable_sampling.py \
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
#     --seed=19
# fi

#---------------------------------------------real-------------------------------------------

# real_character=1

# if [ ${real_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/chilloutmix_0.7/merge_all_chilloutmix/combined_model_base"
#   # fused_model="experiments/composed_edlora/chilloutmix/dog_pet_cat1_dog6_chilloutmix/combined_model_base"
#   expdir="merge_all_chilloutmix"

#   keypose_condition=''
#   keypose_adaptor_weight=1.0

#   sketch_condition='datasets/validation_spatial_condition/multi-objects/dogA_catA_dogB.jpg'
#   sketch_adaptor_weight=1.0

#   context_prompt='three objects in the forest, 4K, high quality, high resolution, best quality'
#   context_neg_prompt='cropped, worst quality, low quality'

#   region1_prompt='[a <dog1> <dog2>, dog, in the forest, 4K, high quality, high resolution, best quality]'
#   region1_neg_prompt="[${context_neg_prompt}]"
#   region1='[160, 76, 505, 350]'

#   region2_prompt='[a <pet_cat11> <pet_cat12>, cat, in the forest, 4K, high quality, high resolution, best quality]'
#   region2_neg_prompt="[${context_neg_prompt}]"
#   region2='[162, 370, 500, 685]'

#   region3_prompt='[a <dog61> <dog62>, dog, in the forest, 4K, high quality, high resolution, best quality]'
#   region3_neg_prompt="[${context_neg_prompt}]"
#   region3='[134, 666, 512, 1005]'

#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"

#   python3 regionally_controlable_sampling.py \
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
#     --seed=14 \
#     --num_image=10
# fi

#---------------------------------------------cat_2 and dog_6-------------------------------------------
# finish mmmb modify
real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix_0_7/cat2_dog6_chilloutmix/combined_model_base"
  expdir="cat2_dog6_chilloutmix"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='final_dataset/merge_mask/cat2_dog6/cat2_dog6_mask.png' # dont care
  sketch_adaptor_weight=1.0

  context_prompt='A <cat21> <cat22> on the right and a <dog61> <dog62> on the left.'
  context_neg_prompt='cropped, worst quality, low quality'

  token1='[<cat21> <cat22>]'
  region1_prompt='[a <cat21> <cat22>, right of image ,high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[58, 227, 500, 487]' # dont care

  token2='[<dog61> <dog62>]'
  region2_prompt='[a <dog61> <dog62>, left of image, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[168, 0, 499, 218]' # dont care

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
  token_rewrite="${token1}|${token2}"

  python3 regionally_controlable_sampling_mmmb.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="results/multi-concept/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=0 \
    --num_image=100 \
    --image_height=512 \
    --image_width=512 \
    --token="${token_rewrite}"
fi

#---------------------------------------------flower_1 and vase-------------------------------------------

# real_character=1

# if [ ${real_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/chilloutmix_1_0/flower_1_vase_chilloutmix/combined_model_base"
#   expdir="flower_1_vase_chilloutmix"

#   keypose_condition=''
#   keypose_adaptor_weight=1.0

#   sketch_condition='final_dataset/merge_mask/flower_1_vase/flower_1_vase_mask.png'
#   sketch_adaptor_weight=1.0

#   context_prompt='A <flower_11> <flower_12> in a <vase1> <vase2>.'
#   context_neg_prompt='cropped, worst quality, low quality'

#   token1='[<flower_11> <flower_12>]'
#   region1_prompt='[a <flower_11> <flower_12>, in a vase, 4K, high quality, high resolution, best quality]'
#   region1_neg_prompt="[${context_neg_prompt}]"
#   region1='[178, 119, 415, 410]'

#   token2='[<vase1> <vase2>]'
#   region2_prompt='[a <vase1> <vase2>, 4K, high quality, high resolution, best quality]'
#   region2_neg_prompt="[${context_neg_prompt}]"
#   region2='[437, 141, 975, 377]'

#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
#   token_rewrite="${token1}|${token2}"

#   python3 regionally_controlable_sampling.py \
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
#     --seed=2 \
#     --num_image=20 \
#     --image_height=512 \
#     --image_width=512 \
#     --token="${token_rewrite}"
# fi

#---------------------------------------------dog_pet_cat1_dog6-------------------------------------------
# finish mmmb modifiy
# real_character=1

# if [ ${real_character} -eq 1 ]
# then
#   fused_model="experiments/composed_edlora/chilloutmix/dog_pet_cat1_dog6_chilloutmix/combined_model_base"
#   expdir="dog_pet_cat1_dog6_chilloutmix"

#   keypose_condition=''
#   keypose_adaptor_weight=1.0

#   sketch_condition='final_dataset/merge_mask/dog_pet_cat1_dog6/dog_pet_cat1_dog6_mask_big.png' # dont care
#   sketch_adaptor_weight=1.0

#   context_prompt='A dog, a cat and a dog near a forest.'
#   context_neg_prompt='cropped, worst quality, low quality, blur'

#   token1='[<dog1> <dog2>]'
#   region1_prompt='[a <dog1> <dog2>, dog, near a forest, 4K, high quality, high resolution, best quality]'
#   region1_neg_prompt="[${context_neg_prompt}]"
#   region1='[50, 5, 398, 345]' # dont care

#   token2='[<pet_cat11> <pet_cat12>]'
#   region2_prompt='[a <pet_cat11> <pet_cat12>, cat, near a forest, 4K, high quality, high resolution, best quality]'
#   region2_neg_prompt="[${context_neg_prompt}]"
#   region2='[47, 360, 400, 683]' # dont care

#   token3='[<dog61> <dog62>]'
#   region3_prompt='[a <dog61> <dog62>, dog, near a forest, 4K, high quality, high resolution, best quality]'
#   region3_neg_prompt="[${context_neg_prompt}]"
#   region3='[46, 704, 400, 1013]' # dont care

#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"
#   token_rewrite="${token1}|${token2}|${token3}"

#   python3 regionally_controlable_sampling_mmmb.py \
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
#     --seed=69 \
#     --num_image=100 \
#     --image_height=512 \
#     --image_width=1024 \
#     --token="${token_rewrite}"
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

#   context_prompt='A <watercolor1> <watercolor2> painting of a <cat21> <cat22> wearing <wearable> <glasses>.'
#   context_neg_prompt='cropped, worst quality, low quality'

#   token1='[<cat21> <cat22>]'
#   region1_prompt='[<watercolor1> <watercolor2> painting of a <cat21> <cat22>, high quality, high resolution, best quality]'
#   region1_neg_prompt="[${context_neg_prompt}]"
#   region1='[21, 48, 488, 435]'

#   token2='[<wearable> <glasses>]'
#   region2_prompt='[<watercolor1> <watercolor2> painting of a <wearable> <glasses>, high quality, high resolution, best quality]'
#   region2_neg_prompt="[${context_neg_prompt}]"
#   region2='[95, 160, 185, 340]'

#   region3_prompt='[<watercolor1> <watercolor2> style]'
#   region3_neg_prompt="[${context_neg_prompt}]"
#   region3='[0, 0, 512, 512]'


#   prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"
#   token_rewrite="${token1}|${token2}"

#   # for non-regionally control
#   global_prompt='A <watercolor1> <watercolor2> painting of a <cat21> <cat22> in a <watercolor1> <watercolor2> style wearing <wearable> <glasses> in a <watercolor1> <watercolor2> style'

#   # prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

#   # use regionally control
#   # python3 regionally_controlable_sampling.py \
#   #   --pretrained_model=${fused_model} \
#   #   --sketch_adaptor_weight=${sketch_adaptor_weight}\
#   #   --sketch_condition=${sketch_condition} \
#   #   --keypose_adaptor_weight=${keypose_adaptor_weight}\
#   #   --keypose_condition=${keypose_condition} \
#   #   --save_dir="results/multi-concept/${expdir}" \
#   #   --prompt="${context_prompt}" \
#   #   --negative_prompt="${context_neg_prompt}" \
#   #   --prompt_rewrite="${prompt_rewrite}" \
#   #   --suffix="baseline" \
#   #   --seed=0 \
#   #   --num_image=20 \
#   #   --image_height=512 \
#   #   --image_width=512 \
#   #   --token="${token_rewrite}"

#   # no use regionally control
#   python3 non_regionally_controlable_sampling.py \
#     --pretrained_model=${fused_model} \
#     --sketch_adaptor_weight=${sketch_adaptor_weight}\
#     --sketch_condition=${sketch_condition} \
#     --keypose_adaptor_weight=${keypose_adaptor_weight}\
#     --keypose_condition=${keypose_condition} \
#     --save_dir="results/multi-concept/${expdir}" \
#     --prompt="${global_prompt}" \
#     --negative_prompt="${context_neg_prompt}" \
#     --prompt_rewrite="${prompt_rewrite}" \
#     --suffix="baseline" \
#     --seed=0 \
#     --num_image=20 \
#     --image_height=512 \
#     --image_width=512 \
#     --token="${token_rewrite}"

# fi

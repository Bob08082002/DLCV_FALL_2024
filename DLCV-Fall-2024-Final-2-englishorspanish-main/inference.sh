
# #---------------------------------------------cat_2 and dog_6-------------------------------------------

real_character=1

if [ ${real_character} -eq 1 ]
then
  #fused_model="experiments/composed_edlora/chilloutmix/cat2_dog6_chilloutmix/combined_model_base"
  fused_model="experiments/composed_edlora/chilloutmix/merge_all_chilloutmix/combined_model_base"
  expdir="0"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='final_dataset/merge_mask/cat2_dog6/cat2_dog6_mask.png'
  sketch_adaptor_weight=1.0

  context_prompt='A <cat21> <cat22> on the right and a <dog61> <dog62> on the left.'
  context_neg_prompt='cropped, worst quality, low quality, blue eyes, open mouth'

  token1='[<cat21> <cat22>]'
  region1_prompt='[a <cat21> <cat22>, cat, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[58, 227, 500, 487]'

  token2='[<dog61> <dog62>]'
  region2_prompt='[a <dog61> <dog62>, dog, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[168, 0, 499, 218]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
  token_rewrite="${token1}|${token2}"

  python3 regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="Generate_image/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3440 \
    --num_image=10 \
    --image_height=512 \
    --image_width=512 \
    --token="${token_rewrite}"
fi

# #---------------------------------------------flower_1 and vase-------------------------------------------

real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/flower_1_vase_chilloutmix/combined_model_base"
  #fused_model="experiments/composed_edlora/chilloutmix/merge_all_chilloutmix/combined_model_base"
  expdir="1"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='final_dataset/merge_mask/flower_1_vase/flower_1_vase_mask.png'
  sketch_adaptor_weight=1.0

  context_prompt='A <flower_11> <flower_12> in a <vase1> <vase2>'
  context_neg_prompt='cropped, worst quality, low quality, complex background'

  token1='[<vase1> <vase2>]'
  region1_prompt='[a <vase1> <vase2>,vase, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[437, 141, 975, 377]'

  token2='[<flower_11> <flower_12>]'
  region2_prompt='[a <flower_11> <flower_12>, flower, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[178, 119, 415, 410]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"
  token_rewrite="${token1}|${token2}"

  global_prompt='A <flower_11> <flower_12> in a <vase1> <vase2>.'

  python3 regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="Generate_image/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=11 \
    --num_image=10 \
    --image_height=1024 \
    --image_width=512 \
    --token="${token_rewrite}"

fi

# # # #---------------------------------------------dog_pet_cat1_dog6-------------------------------------------

real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/dog_pet_cat1_dog6_chilloutmix/combined_model_base"
  #fused_model="experiments/composed_edlora/chilloutmix/merge_all_chilloutmix/combined_model_base"
  expdir="2"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='final_dataset/merge_mask/dog_pet_cat1_dog6/dog_pet_cat1_dog6_mask.png'
  sketch_adaptor_weight=1.0

  context_prompt='A dag, a cat and a dag near a forest.'
  context_neg_prompt='cropped, worst quality, low quality, blur'

  token1='[<dog1> <dog2>]'
  region1_prompt='[a <dog1> <dog2>, dog, near a forest, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[130, 27, 482, 357]'

  token2='[<pet_cat11> <pet_cat12>]'
  region2_prompt='[a <pet_cat11> <pet_cat12>, cat, near a forest, 4K, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[128, 387, 484, 709]'

  token3='[<dog61> <dog62>]'
  region3_prompt='[a <dog61> <dog62>, dog, near a forest, 4K, high quality, high resolution, best quality]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[125, 730, 485, 1011]'

  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"
  token_rewrite="${token1}|${token2}|${token3}"

  python3 regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="Generate_image/${expdir}" \
    --prompt="${context_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=12 \
    --num_image=10 \
    --image_height=512 \
    --image_width=1024 \
    --token="${token_rewrite}"
fi

# # # #---------------------------------------------cat2 wearable_glasses watercolor-------------------------------------------

real_character=1

if [ ${real_character} -eq 1 ]
then
  fused_model="experiments/composed_edlora/chilloutmix/cat2_wearable_glasses_watercolor_chilloutmix/combined_model_base"
  #fused_model="experiments/composed_edlora/chilloutmix/merge_all_chilloutmix/combined_model_base"
  expdir="3"

  keypose_condition=''
  keypose_adaptor_weight=1.0

  sketch_condition='final_dataset/merge_mask/cat2_wearable_glasses_watercolor/cat2_wearable_glasses_watercolor_mask.png'
  sketch_adaptor_weight=1.0

  context_prompt='A <cat21> <cat22> wearing <wearable> <glasses> in a <watercolor1> <watercolor2> style.'
  context_neg_prompt='half body, cropped, low quality, blur'

  token1='[<cat21> <cat22>]'
  region1_prompt='[<watercolor1> <watercolor2> painting of a <cat21> <cat22>, high quality, high resolution, best quality]'
  region1_neg_prompt="[${context_neg_prompt}]"
  region1='[21, 48, 488, 435]'

  token2='[<wearable> <glasses>]'
  region2_prompt='[<watercolor1> <watercolor2> painting of a <wearable> <glasses>, high quality, high resolution, best quality]'
  region2_neg_prompt="[${context_neg_prompt}]"
  region2='[95, 160, 185, 340]'

  region3_prompt='[<watercolor1> <watercolor2> style]'
  region3_neg_prompt="[${context_neg_prompt}]"
  region3='[0, 0, 512, 512]'


  prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}|${region3_prompt}-*-${region3_neg_prompt}-*-${region3}"
  token_rewrite="${token1}|${token2}"

  # for non-regionally control
  global_prompt='A <watercolor1> <watercolor2> painting of a <cat21> <cat22> in a <watercolor1> <watercolor2> style wearing <wearable> <glasses> of <watercolor1> <watercolor2> style.'

  # prompt_rewrite="${region1_prompt}-*-${region1_neg_prompt}-*-${region1}|${region2_prompt}-*-${region2_neg_prompt}-*-${region2}"

  # use regionally control
  # python3 regionally_controlable_sampling.py \
  #   --pretrained_model=${fused_model} \
  #   --sketch_adaptor_weight=${sketch_adaptor_weight}\
  #   --sketch_condition=${sketch_condition} \
  #   --keypose_adaptor_weight=${keypose_adaptor_weight}\
  #   --keypose_condition=${keypose_condition} \
  #   --save_dir="Generate_image_v6/${expdir}" \
  #   --prompt="${context_prompt}" \
  #   --negative_prompt="${context_neg_prompt}" \
  #   --prompt_rewrite="${prompt_rewrite}" \
  #   --suffix="baseline" \
  #   --seed=0 \
  #   --num_image=10 \
  #   --image_height=512 \
  #   --image_width=512 \
  #   --token="${token_rewrite}"

  # no use regionally control
  python3 non_regionally_controlable_sampling.py \
    --pretrained_model=${fused_model} \
    --sketch_adaptor_weight=${sketch_adaptor_weight}\
    --sketch_condition=${sketch_condition} \
    --keypose_adaptor_weight=${keypose_adaptor_weight}\
    --keypose_condition=${keypose_condition} \
    --save_dir="Generate_image/${expdir}" \
    --prompt="${global_prompt}" \
    --negative_prompt="${context_neg_prompt}" \
    --prompt_rewrite="${prompt_rewrite}" \
    --suffix="baseline" \
    --seed=3407 \
    --num_image=10 \
    --image_height=512 \
    --image_width=512 \
    --token="${token_rewrite}"

fi

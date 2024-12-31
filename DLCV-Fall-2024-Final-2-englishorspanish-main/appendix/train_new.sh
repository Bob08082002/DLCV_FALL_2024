# dog
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/dog.yml
# cat2
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/cat2.yml
# dog6
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/dog6.yml
# flower_1
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/flower_1.yml
# pet_cat1
# accelerate launch train_edlora.py -opt options/train/EDLoRA/real/pet_cat1.yml

# For train_edlora.py
# vase
python3 train_edlora_new.py -opt config_new/train/EDLoRA/real/vase.yml --task_id 1
# wearable_glasses
python3 train_edlora_new.py -opt config_new/train/EDLoRA/real/wearable_glasses.yml --task_id 2
# watercolor
python3 train_edlora_new.py -opt config_new/train/EDLoRA/real/watercolor.yml --task_id 3

# For train_edlora_new.py

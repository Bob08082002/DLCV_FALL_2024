# DLCV final project - Multiple concept personalization | team5 EnglishOrSpanish

## Environment Setup
To set up the required environment, please refer to the environment_setup.cmd file. Our codebase integrates functionalities from Mix-of-Show, GroundingDINO, and Segment Anything. The environment merges dependencies and configurations from these works.

```bash
source environment_setup.cmd
```
Please note that this step may require some time.
If you encounter any issues while following the installation process outlined in environment_setup.cmd, please don't hesitate to reach out to us for assistance.

## Prepared Dataset
Ensure that all the files are organized in the correct hierarchy before proceeding.
### Pre-trained Model
Download all the pre-trained models using the following command:

```bash
bash download_ckpt.sh
```
Please note that this step may require some time.
After unzipping, verify that the experiments folder exists and has the following structure:

```bash
experiments/
    cat2/
    composed_edlora/
    dog/
    dog6/
    flower_1/
    pet_cat1/
    pretrained_models/
    vase/
    watercolor/
    wearable_glasses/
```
These pre-trained models include the fused ED-LoRA, allowing you to perform inference directly using these models.
### For Single-concept Training
All files required for single-concept training are located in the final_dataset folder. The folder structure is as follows:

```bash
final_dataset/
    captions/
    concept_image/
    configs/
    mask/
    merge_mask/
    prompts.json
```

captions/: Contains the prompts used for validation during training.

concept_image/: Stores the source images for training.

configs/: Contains the training configuration files.

mask/: Holds the masks used for training.

merge_mask/: Includes merged mask files used for inference.

The detailed training configurations for single-concept training are located in the options folder.
```bash
options/train/EDLoRA/real/
    cat2.yml
    dag.yml
    dog6.yml
    flower_1.yml
    pet_cat1.yml
    vase.yml
    watercolor.yml
    wearable_glasses.yml
```
To train a single concept using ED-LoRA, use the train.sh shell script.

### For multiple concept ED-LoRA Fused
Ensure that the following five configuration files are located in the datasets/data_cfgs/ directory:
```bash
datasets/data_cfgs/cat2_dog6_chilloutmix.json
datasets/data_cfgs/cat2_wearable_glasses_watercolor_chilloutmix.json
datasets/data_cfgs/dog_pet_cat1_dog6_chilloutmix.json
datasets/data_cfgs/flower_1_vase_chilloutmix.json
datasets/data_cfgs/merge_all_chilloutmix.json
```
To fuse multiple concept ED-LoRA, execute the fuse.sh shell script.
### For multiple concept Inference
Use the inference.sh shell script to generate images based on the provided prompts.

### Important Notes
Ensure all pre-trained models are downloaded and placed in their correct locations.
If the file paths in inference.sh are incorrect, you can manually update them as needed.

## Train ED-Lora for single concept
To train the single-concept ED-LoRA, simply run the following command:

```bash
bash train.sh
```
You can customize the training settings by modifying the files under options/train/EDLoRA/real/*.yml to suit your requirements.
## Fuse single concept into multiple concept
To fuse the single-concept ED-LoRA, run the following command:

```bash
bash fuse.sh
```
You can customize the fusing settings by editing the configuration files located under datasets/data_cfgs/*.json to match your requirements.
## Inference
To reproduce the results of our work, use the following command:

```bash
bash inference.sh
```

For consistent results, such as those on Codalab, ensure that:
The required pre-trained models remain unmodified.
All configuration files are unchanged from those provided in the GitHub repository.
## GUI for User-Defined Bounding Box Generation
We provide a GUI tool for generating user-defined bounding boxes. To ensure the GUI works properly, please use MobaXterm.

Run the following command to start the GUI:
```bash
bash pipeline_test.sh
```
After generating the bounding box:

The mask will be saved to the path specified in the shell script.
The bounding box information will be saved in the same directory as a file named 0.txt.
If you want to perform inference on the images, update the bounding box information in inference.sh with the newly generated bounding box data.

### Note on the Pipeline
While we aimed to streamline the entire process into a single script, GPU memory limitations required us to separate the pipeline into two stages. We apologize for any inconvenience this may cause.

## Appendix
We provide some supplementary code related to our work, though its use is not mandatory. Most of the files in the appendix folder are for experimental purposes and exploratory trials.
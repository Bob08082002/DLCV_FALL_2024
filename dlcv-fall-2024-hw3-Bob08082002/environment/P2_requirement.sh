# Create a new conda environment
#conda create --name DLCV_HW3_P2_ENV python=3.8
# Activate the environment
#conda activate DLCV_HW3_P2_ENV

# Install PyTorch and torchvision
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
#pip install torch==1.12.1
#pip install torchvision==0.13.1

# Install packages using pip 
pip install imageio==2.21.3
pip install matplotlib==3.6.1
pip install numpy==1.23.4
pip install Pillow==9.2.0
pip install scipy==1.9.1
pip install opencv-python==4.6.0.66
pip install loralib==0.1.2
pip install pycocotools==2.0.5
pip install timm==0.9.10
pip install pandas==1.5.1
pip install tqdm
#pip install gdown
#pip install glob
#pip install yaml
#pip install skimage


# Install package language-evaluation
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"

# Install package CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
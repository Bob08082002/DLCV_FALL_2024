# Create a new conda environment
#conda create --name DLCV_HW3_P1_ENV python=3.8
# Activate the environment
#conda activate DLCV_HW3_P1_ENV

# Install PyTorch and torchvision
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install bitsandbytes, accelerate, and transformers using pip 
pip install bitsandbytes==0.44.1
pip install accelerate==1.0.1
pip install transformers==4.45.2

# Install package language-evaluation
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"

# Install package CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
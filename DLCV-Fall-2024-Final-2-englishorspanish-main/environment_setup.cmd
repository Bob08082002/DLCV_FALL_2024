conda create -n EnglishOrSpanish python=3.10 -y
conda activate EnglishOrSpanish

# GROUNDING DINO

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../

# SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# for dino
pip install numpy==1.26.4

# MIX OF SHOW
pip install diffusers==0.28.0 
pip install huggingface-hub==0.24.6
pip install einops==0.8.0 

#for google drive download
pip install gdown

# train
pip install accelerate
pip install transformers==4.26.1

# re-install for lower version
pip install diffusers==0.19.3 

# for training
pip install omegaconf IPython

#for PyQt (mask GUI)
pip install PyQt5==5.15.9 pyqt5-tools==5.15.9.3.3
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install opencv-python-headless

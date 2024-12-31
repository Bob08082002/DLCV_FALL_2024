git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

# --------------- Caution: comment pip submoudles in environment.yml --------------- #
conda env create --file environment.yml
conda activate gaussian_splatting
conda install -c conda-forge cudatoolkit-dev

# pip submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim


# for grade.py
pip install scipy==1.7.3
pip install imageio==2.31.2
pip install lpips==0.1.4
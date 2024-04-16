# /bin/bash

source $CONDA_PREFIX/etc/profile.d/conda.sh

# Delete if already exists
conda env remove -n motion2vecsets -y

conda create -n motion2vecsets python=3.7 -y
conda activate motion2vecsets 
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit==11.3 -c pytorch 

pip install https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_cluster-1.6.0-cp37-cp37m-linux_x86_64.whl
pip install -r script/requirements.txt

python ./setup_im2mesh.py build_ext --inplace
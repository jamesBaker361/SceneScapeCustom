conda init
eval "$(conda shell.bash hook)"
conda deactivate
conda remove -n scenescape --all -y
pip cache purge
conda clean --all -y
conda create --name scenescape python=3.10 -y
conda activate scenescape 
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
conda install -c conda-forge libgl
pip install git+https://github.com/omry/omegaconf.git
pip install -r requirements.txt
pip install experiment-helpers
python run.py
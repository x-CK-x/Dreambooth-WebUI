pip install omegaconf
pip install einops
pip install test-tube
pip install transformers
pip install kornia
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install setuptools==59.5.0
pip install pillow==9.0.1
pip install torchmetrics==0.6.0
pip install -e .
pip install protobuf==3.20.1
pip install gdown
pip install pydrive
pip install -qq diffusers["training"]==0.3.0 transformers ftfy
pip install -qq "ipywidgets>=7,<8"
pip install huggingface_hub
pip install ipywidgets==7.7.1
pip install gradio
pip install torchvision
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install pytorch-lightning==1.7.6
pip install clip
export MKL_SERVICE_FORCE_INTEL=1
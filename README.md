# Dreambooth Stable Diffusion Demo w/ a GUI ::: Based on [JoePenna's Implementation](https://github.com/JoePenna/Dreambooth-Stable-Diffusion) 🤗
### Setup/Configure/Regularization/Fine-Tune: with this DreamBooth stable diffusion demo.

## BEFORE continuing. Please Verify the following pre-requisites:
    - the system is running a NVIDIA gpu
    - the gpu has ~24 GBs for fine-tuning
    - python is installed
    - NVIDIA drivers are up to date
    - the system has a compatible version of the CUDA toolkit installed
    - the system has a compatible version of the cuDNN toolkit installed

Please make sure to install python: [Python Official Website](https://www.python.org/downloads/)
1. And add it to PATH
2. Install the packages with:
    - ./setup.sh
3. Run the UI:  python webui.py
4. Follow the 1-Time installation setup on the GUI
- Features a Model & Dataset Configuration tab, and Model Training tab


### Additional planned features:
- [ ] The Pruning Feature from the repo (12GB to 2GB)
- [ ] Resolve Gradio Components not updating Components for real-time logging feed
- [ ] A Training Job Scheduler
- [ ] An Advanced Image Cropping Tool using YOLOv7
- [ ] Multi-GPU support:
    - Model Parallelism
- [ ] Utilizing Other Sampling Algorithms

# GRU-Based Neural Decoding
Official repository for the paper ["**Optimized AI-based neural decoding from BOLD fMRI signal for analyzing visual and semantic ROIs in the human visual system**"](https://iopscience.iop.org/article/10.1088/1741-2552/adfbc2/meta) by Lorenzo Veronese et al. published on Journal of Neural Engineering.

## Results
The following are a few reconstructions obtained : 
<p align="center"><img src="./figures/Results.png" width="600" ></p>

## Instructions 

### Requirements
* Create conda environment using environment.yml in the main directory by entering `conda env create -f environment.yml`.

### Data and Models Acquisition 
0. For data and model acquisition, we follow the pipeline proposed by Furkan Ozcelik and Rufin VanRullen (https://github.com/ozcelikfu/brain-diffuser)
1. Download NSD data from NSD AWS Server. While in the root folder, run:
    ```
	python download_data.py
	```
2. Download "COCO_73k_annots_curated.npy" file from [HuggingFace NSD](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main)

3. Download pretrained VDVAE model files and put them in `vdvae/model/` folder
	```
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
	```

4. Download pretrained Versatile Diffusion model "vd-four-flow-v1-0-fp16-deprecated.pth", "kl-f8.pth" and "optimus-vae.pth" from [HuggingFace](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth) and put them in `versatile_diffusion/pretrained/` folder

### Preprocessing and Neural Decoding

* While in the root folder, run:
	```
	run_all_commands.py
	```
	In this convenient code, parameters of the model can be set, steps of the decoding can be selected, and data can be stored for further analysis.

## References
- Codes of our neural decoding pipeline are an evolution of [brain-diffuser] (https://github.com/ozcelikfu/brain-diffuser)
- Codes in vdvae directory are derived from [openai/vdvae](https://github.com/openai/vdvae)
- Codes in versatile_diffusion directory are derived from earlier version of [SHI-Labs/Versatile-Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion)
- Dataset used in the studies is obtained from [Natural Scenes Dataset](https://naturalscenesdataset.org/)

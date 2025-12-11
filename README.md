# GRU-Based Neural Decoding
Official repository for the paper ["**Optimized AI-based neural decoding from BOLD fMRI signal for analyzing visual and semantic ROIs in the human visual system**"](https://iopscience.iop.org/article/10.1088/1741-2552/adfbc2/meta) by Lorenzo Veronese et al. published on Journal of Neural Engineering.

<p align="center"><img src="./figures/Methods.png" width="600" ></p>

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
	```
	https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/COCO_73k_annots_curated.npy?download=true
 	```
 
3. Download pretrained VDVAE model files and put them in `vdvae/model/` folder
	```
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
	wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
	```

4. Download pretrained Versatile Diffusion model "vd-four-flow-v1-0-fp16-deprecated.pth", "kl-f8.pth" and "optimus-vae.pth" from [HuggingFace](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth) and put them in `versatile_diffusion/pretrained/` folder
   ```
	wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/vd-four-flow-v1-0-fp16-deprecated.pth?download=true
	wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/kl-f8.pth?download=true
	wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/optimus-vae.pth?download=true
	```

### Preprocessing and Neural Decoding

* To execute the full pipeline, run the following command from the root directory:
	```
	run_all_commands.py
	```
	This script serves as the main entry point for the project. It allows you to configure model hyperparameters, select specific decoding stages (e.g., Stage 1 vs. Stage 2), and save output data for quantitative analysis.

## References
- Codes of our neural decoding pipeline are an evolution of [brain-diffuser](https://github.com/ozcelikfu/brain-diffuser)
- Dataset used is the [Natural Scenes Dataset](https://naturalscenesdataset.org/)
- Codes in vdvae directory are derived from [openai/vdvae](https://github.com/openai/vdvae)
- Codes in versatile_diffusion directory are derived from [SHI-Labs/Versatile-Diffusion](https://github.com/SHI-Labs/Versatile-Diffusion)

## Citation
If you find this work helpful, please consider citing our paper:

```bibtex
@Article{Veronese2025,
  author    = {Veronese, Lorenzo and Moglia, Andrea and Pecco, Nicolò and Anthony Della Rosa, Pasquale and Scifo, Paola and Mainardi, Luca and Cerveri, Pietro},
  journal   = {Journal of Neural Engineering},
  title     = {Optimized AI-based neural decoding from BOLD fMRI signal for analyzing visual and semantic ROIs in the human visual system},
  year      = {2025},
  issn      = {1741-2552},
  month     = aug,
  number    = {4},
  pages     = {046048},
  volume    = {22},
  doi       = {10.1088/1741-2552/adfbc2},
  publisher = {IOP Publishing},
}
```

## Authors
* **[Lorenzo Veronese](https://www.linkedin.com/in/lorenzo-v-veronese):** Conceived the study, developed the methodology, implemented experiments, performed analysis, and edited the manuscript.
* **Andrea Moglia:** Assisted in paper review.
* **[Nicolò Pecco](https://www.linkedin.com/in/nicolo-pecco/):** Participated in data analysis and synthesis, and reviewed the manuscript.
* **Pasquale Anthony Della Rosa:** Provided insights on cognitive neuroscience and fMRI imaging, and reviewed the manuscript.
* **Paola Scifo:** Provided insights on fMRI imaging.
* **Luca Mainardi:** Assisted in the paper reviewing process.
* **Pietro Cerveri:** Conceived the study, supervised the work, and provided financial support.

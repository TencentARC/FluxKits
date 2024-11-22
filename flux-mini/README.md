# Flux-Mini
A distilled Flux-dev model for efficient text-to-image generation

## Distillation Strategy

Our goal is to distill the  `12B Flux-dev` into a `3.2B Flux-Mini` model, by pruning the blocks in the original model from 19 double blocks and 38 single blocks into 5 double blocks and 10 single blocks. 
We empeirically found that different blocks has different impact on the generation quality, thus we initialize the student model with several most important blocks. The distillation process consists of three objectives: the denoise loss, the output alignment loss and the feature alignment loss. The feature aligement loss is designed in a way such that the output of `block-x` in the student model is encouraged to match that of `block-4x` in the teacher model. 
The distillation process is performed with `512x512` laion images recaptioned with `Qwen-VL` in the first stage for `90k steps`, and `1024x1024` images generated by `Flux` using the prompts in `JourneyDB` with another `90k steps`.


## Environment Setup

```bash
  git clone https://github.com/TencentARC/FluxKits
  cd FluxKits/flux-mini
  conda create -n flux-mini python==3.11.9
  pip install -r requirements.txt
```

## Usage

### Downloading Weights

To run the code with our Flux-mini, one need to get the following model weights:
* [Flux-mini](https://huggingface.co/TencentARC/flux-mini)
* [Flux-Autoencoder](https://huggingface.co/black-forest-labs/FLUX.1-dev)
* [T5-xxl](https://huggingface.co/google/t5-v1_1-xxl)
* [CLIP](https://huggingface.co/openai/clip-vit-large-patch14)

> The weights of the above model will be automatically downloaded from HuggingFace once you start one of the demos. 

> You may also download the previous weights manually using `python` with  `huggingface_hub`. 
```python
from huggingface_hub import hf_hub_download, snapshot_download
FLUX_MINI_PATH = hf_hub_download(repo_id="TencentARC/flux-mini", filename="flux-mini.safetensors", repo_type="model")
AE_PATH = hf_hub_download(repo_id="black-forest-labs/FLUX.1-schnell", filename="ae.safetensors", repo_type="model")
GOOGLE_PATH = snapshot_download(repo_id='google/t5-v1_1-xxl')
OPENAI_PATH = snapshot_download(repo_id='openai/clip-vit-large-patch14')
```

Set the path of the checkpoints. Replace the `XX_PATH` below with the corresponding values returned above.

```shell
cd FluxKits/flux-mini/src
ln -s GOOGE_PATH .
ln -s OPENAI_PATH .
export FLUX_MINI=FLUX_MINI_PATH
export AE=AE_PATH
```

### Generating images



For interactive sampling, run
```python
python -m flux --name <name> --loop
```

Or generate a single sample with
```python
python -m flux --name <name> --height <height> --width <width> --prompt "<prompt>"
```

## Training


### Dataset Preparation
The model could be trained on image-text datasets. The dataset has the following format for the training process:

```text
├── images/
│    ├── 1.png
│    ├── 1.txt
│    ├── 2.png
│    ├── 2.txt
│    ├── ...
```


### Set dataset and model path

#### Dataset Path
Change the path of the dataset in `train_configs/*.yaml`
```
data_config:
  train_batch_size: 1
  num_workers: 1
  img_size: 1024
  img_dir:  /path/to/dataset # your dataset location
```

#### Model Path
```bash
cd FluxKits/flux-mini
ln -s GOOGLE_PATH .
ln -s OPENAI_PATH .
```

### Start Training

Train flux-mini with LoRA:
```
./scripts/train_lora.sh
```


We also provide codes to run the model distillation:

```
./scripts/train_distill.sh
```
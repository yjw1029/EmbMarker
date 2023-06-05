# EmbMarker
Code and data for our paper "Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark" in ACL 2023.

## Introduction


## Environment

### Docker

We suggest docker to manage enviroments. You can pull the pre-built image from docker hub
```bash
docker pull yjw1029/torch:1.13.0
```
or build the image by yourself
```
docker build -f Dockerfile -t yjw1029/torch:1.13.0 .
```

### conda or pip
You can also install required packages with conda or pip.
The package requirements are as follows
```
accelerate>=0.12.0
wandb
transformers==4.25.1
evaluate==0.3.0
datasets
torch==1.13.0
numpy
tqdm

# if you want to request embeddings from openai api
openai
```


## Getting Started

### Preparing dataset
We directly use the SST2, Enron Spam and AG News published on huggingface datasets.
For MIND datasets, we merge the all news in its recommendation logs and split in to train and test files.
You can download [here](https://drive.google.com/file/d/19kO8Yy2eVLzSL0DFrQ__BHjKyHUoQf6R/view?usp=drive_link) for train and [here](https://drive.google.com/file/d/1O3KTWhfnqxmqPNFChGR-bv8rAv-mzLQZ/view?usp=drive_link) for testing.

### Requesting GPT3 Embeddings
We release the pre-requested embeddings. You can click the link to download them one by one into data directory.
| dataset | split | download link |
|  --     |   --  |      --       |
|  SST2   | train |  [link](https://drive.google.com/file/d/1JnBlJS6_VYZM2tCwgQ9ujFA-nKS8-4lr/view?usp=drive_link)     |
|  SST2   | validation | [link](https://drive.google.com/file/d/1-0atDfWSwrpTVwxNAfZDp7VCN8xQSfX3/view?usp=drive_link) |
|  SST2   | test  |  [link](https://drive.google.com/file/d/157koMoB9Kbks_zfTC8T9oT9pjXFYluKa/view?usp=drive_link)     |
|  Enron Spam | train | [link](https://drive.google.com/file/d/1N6vpDBPoHdzkH2SFWPmg4bzVglzmhCMY/view?usp=drive_link)  |
|  Enron Spam | test  | [link](https://drive.google.com/file/d/1LrTFnTKkNDs6FHvQLfmZOTZRUb2Yq0oW/view?usp=drive_link)  |
|  Ag News | train | [link](https://drive.google.com/file/d/1r921scZt8Zd8Lj-i_i65aNiHka98nk34/view?usp=drive_link) |
|  Ag News | test  | [link](https://drive.google.com/file/d/1adpi7n-_gagQ1BULLNsHoUbb0zbb-kX6/view?usp=drive_link) |
|  MIND    | all | [link](https://drive.google.com/file/d/1pq_1kIe2zqwZAhHuROtO-DX_c36__e7J/view?usp=drive_link) |


Or download the embddings and MIND news files via our script based on [gdown](https://github.com/wkentaro/gdown).
```bash
pip install gdown
bash preparation/download_emb.sh
```


Since there exists randomness in OpenAI embedding API, we recommend you to use our released embeddings for experiment reporduction.
You can also request the embeddings by yourselves.

```bash
cd preparation
python request_emb.py
```

### Counting word frequency
The pre-computed word count file is [here](https://drive.google.com/file/d/1YrSkDoQL7ComIBr7wYkl1muqZsWSYC2t/view?usp=drive_link).
You can also preprocess wikitext dataset to get the same file.
```bash
export OPENAI_API_KEYS="YOUR API KEY"
cd preparation
python word_count.py
```

### Run Experiments
```
```


## Results

## Citing

```latex
@article{peng2023you,
  title={Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark},
  author={Peng, Wenjun and Yi, Jingwei and Wu, Fangzhao and Wu, Shangxi and Zhu, Bin and Lyu, Lingjuan and Jiao, Binxing and Xu, Tong and Sun, Guangzhong and Xie, Xing},
  journal={arXiv preprint arXiv:2305.10036},
  year={2023}
}
```
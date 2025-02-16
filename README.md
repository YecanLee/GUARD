## GUARD: Glocal Uncertainty Aware Robust Decoding <br><sub>Official PyTorch Implementation</sub>
## [Paper](placeholder) | [Project Page](placeholder) | Run Analysis Baseline [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](placeholder)

This repo contains the official implementation of our paper __"GUARD: Glocal Uncertainty Aware Robust Decoding"__. You can find more details in our [project page](placeholder) and our [paper](placeholder).

> [**GUARD: Glocal Uncertainty Aware Robust Decoding**](placeholder)<br>


<p align="center">
  <img src="assets/cs_trend.jpeg" alt="Contrastive Search Trend">
</p>


## 📅 Timeline
- [2025/02/16] We have released the official code implementation of our paper 🤩.


## 📖 Table of Contents <a href="#top">[Back to Top]</a>

- [Download Pre-generated Dataset](#Download-Pre-generated-Dataset-)
- [Dependency Installation](#Dependency-installation-)
- [Run Paper Inference Experiments](#Run-Paper-Inference-Experiments-)
- [Run Benchmark Inference Experiments](#Run-Benchmark-Inference-Experiments-)
- [Benchmark Decoding Methods](#Benchmark-Decoding-Methods-)
  - [Measure Diversity, Generation Length and MAUVE Score](#Measure-Diversity,-Generation-Length-and-MAUVE-Score-)
- [Enhancements](#Enhancements-)
- [License](#License-)
- [Contributions](#Contributions-)

## 🌠 Download Pre-generated Dataset <a href="#top">[Back to Top]</a> <a name="download-pre-generated-dataset-"></a>
To download the pre-generated dataset used for model comparison in our paper, please run the following command:
```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Xa1ZtZpqL7bySVEy_Q8fqGjfNN7L-xvG
```


## 🛸 Dependency Installation <a href="#top">[Back to Top]</a> <a name="dependency-installation-"></a>

To install all the dependencies for our paper, run the following command:
```bash
pip install -r requirements.txt
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install simctg
```

We recommend you to build a new conda environment to use the repository.

```bash
conda create -n guard python=3.11
conda activate guard
pip install -r requirements.txt

# install `simctg` for metric calculation
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install simctg
```

## 🚀 Run Paper Inference Experiments <a href="#top">[Back to Top]</a> <a name="run-paper-inference-experiments-"></a>
You could choose to run the inference experiments for our proposed method by using one of the following ways:

### Run with huggingface transformers library
To run the inference experiments for our proposed method by using the huggingface transformers library, please run the following command:
```bash
conda activate guard
pip uninstall transformers
cd transformers
pip install -e .
cd ..

python llm_exp/llm_iacs.py \
--dataset wikitext \
--q 1.0 \
--temp 0.9 \
--save_file iacs \
--save_path_prefix IACS \
--window_size 7 \
--model_name mistralai/Mistral-7B-v0.3 \
```

### Run with our provided python script
You can also choose to run with a simpler way by using the python script we offered here, notice that the temperature and q initial value are fixed to 0.9 and 1.0 respectively:
```bash
python story_generate.py \
--model_name mistralai/Mistral-7B-v0.3 \
--window_size 7 \
--dataset_name wikitext \
```

## 🚀 Run Benchmark Inference Experiments <a href="#top">[Back to Top]</a> <a name="run-benchmark-inference-experiments-"></a>

We compared 5 different decoding methods with our proposed method in our paper, those are: **[Contrastive Search](https://arxiv.org/abs/2210.14140), [Top-k Sampling](https://arxiv.org/pdf/1805.04833), [Top-p Sampling](https://arxiv.org/abs/1904.09751), [Beam Search](https://arxiv.org/abs/1702.01806) and [Temperature Scaling](https://arxiv.org/abs/1706.04599)**. We compare those methods with the following hyperparameter combinations:
- **Contrastive Search**: alpha=0.6, k=10
- **Top-k Sampling**: k=10
- **Top-p Sampling**: p=0.9
- **Temperature Scaling**: temperature=0.9

We run the decoding methods on the following 6 models:
- [Llama-2](https://www.llama.com/)
- [Llama-3.1](https://www.llama.com/)
- [GPT-2](https://openai.com/index/better-language-models/)
- [Mistral-7B-v0.3](https://mistral.ai/)
- [Qwen/Qwen2.5-7B](https://arxiv.org/abs/2407.10671)
- [deepseek-ai/deepseek-llm-7b-base](https://arxiv.org/abs/2407.14885)


We then benchmark the decoding quality and perplexity of those decoding methods. 

We used the pre-generated dataset for model comparison in our paper to run the experiments.

If you want to generate the pre-generated dataset for your own models, you may need to authorize by logging in to the Hugging Face to run the experiments for Llama-3.1 and Mistral-7B-v0.3.

```bash
huggingface-cli login
```

To run the LLM inference experiments for contrastive search decoding method, run the following command:
```bash
python llm_exp/llm_contrastive_search.py \
--dataset wikitext \
--k 20 \
--alpha 0.8 \
--save_file misrtalv03 \
--save_path_prefix Mistralv03-alpha08 \
--model_name mistralai/Mistral-7B-v0.3 \
--cuda 0 \
--dataset_prefix ./data
```

To run the LLM inference experiments for top-k sampling decoding method, run the following command:
```bash
python llm_exp/llm_top-k.py \
--k 20 \
--save_file gpt2-xl \
--save_path_prefix GPT2-XL-topk \
--dataset wikitext \
--model_name openai-community/gpt2-xl \
--cuda 0 \
```

To run the LLM inference experiments for top-p sampling decoding method, run the following command:
```bash
python llm_exp/llm_top-p.py \
--p 0.95 \
--save_file qwen2_5 \
--save_path_prefix Qwen2.5-7B-topp \
--dataset wikitext \
--model_name Qwen/Qwen2.5-7B \
--cuda 0 \
```

To run the LLM inference experiments for temperature scaling decoding method, run the following command:

```bash
python llm_exp/llm_temp.py \
--temp 0.1 \
--save_file mistralv03 \
--dataset wikitext \
--model_name mistralai/Mistral-7B-v0.3 \
--save_path_prefix mistralv03-temp \
--cuda 0 \
```

## 🧪 Benchmark Decoding Methods <a href="#top">[Back to Top]</a> <a name="benchmark-decoding-methods-"></a>

To benchmark the decoding methods, please make sure you have all the dependencies installed.

We provide scripts for measuring the diversity, generation length and MAUVE score of the generated texts. 

### 🧪 Measure Diversity, Generation Length and MAUVE Score <a href="#top">[Back to Top]</a> <a name="measure-diversity,-generation-length-and-mauve-score-"></a>

#### Measure Diversity, Generation Length and MAUVE Score for a single generated text file
To measure the diversity, generation length and MAUVE score of the generated texts for a single generated text file, please run the following command:

```bash
# change the test path to the file path you want to evaluate
bash scripts/measure_single_mauve.sh YOUR_TEST_PATH
bash scripts/measure_single_coherence.sh YOUR_TEST_PATH
```

## 💪 Enhancements <a href="#top">[Back to Top]</a> <a name="enhancements-"></a>
Generation could likely be speed-up by:
- [x] using `torch.compile` in PyTorch 2.0, we implemented this by using `max_autotune` mode in the generation scripts, you may need to modify the `torch.compile` codes to fit your needs.  

**TF32 Note (important for Ampere, Hopper, and other recent NVIDIA GPUs users).**    
When we ran the above generation scripts, TF32 matmuls were disabled per PyTorch's defaults.    
We've enabled them at the top of `measure_CD_mauve_diversity_gen_len.py` and `measure_diversity_mauve_gen_length.py` because it makes sampling way way way faster on 
those GPUs, but note that the use of TF32 may lead to some differences in the results. Those differences are likely to be negligible for most comparison purposes.

## 📄 License
See [`LICENSE.txt`](LICENSE.txt) for details.

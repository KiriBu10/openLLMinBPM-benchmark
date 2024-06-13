# Towards a Benchmark for Large Language Models for Business Process Management Tasks
This repository contains the code for the paper "Towards a Benchmark for Large Language Models for Business Process Management Tasks".

## About the project
An increasing number of organizations are deploying Large Language Models (LLMs) for a wide range of tasks. Despite their general utility, LLMs are prone to errors, ranging from inaccuracies to hallucinations. To objectively assess the capabilities of existing LLMs, performance benchmarks are conducted. However, these benchmarks often do not translate to more specific real-world tasks. This paper addresses the gap in benchmarking LLM performance in the Business Process Management (BPM) domain. Currently, no BPM-specific benchmarks exist, creating uncertainty about the suitability of different LLMs for BPM tasks. This paper systematically compares LLM performance on four BPM tasks focusing on small open-source models.
The analysis aims to identify task-specific performance variations, compare the effectiveness of open-source versus commercial models, and assess the impact of model size on BPM task performance. This paper provides insights into the practical applications of LLMs in BPM, guiding organizations in selecting appropriate models for their specific needs.

### Built with
* ![Static Badge](https://img.shields.io/badge/Plattform-Windows-blue)
* ![Static Badge](https://img.shields.io/badge/GPU-Nvidia%20RTX%20A6000-red)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.8.13)

## Project Organization
    ├── utils                                      <- Utils folder.
    │   ├ metrics.py                               <- Contains metric functions.
    │   └ preprocessing.py                         <- Contains preprocessing functions. 
    ├── data                                       <- Source folder to datasets.
    │   └ ...                                      <- Datasets.
    ├── evaluation.ipynb                           <- Evaluation script.
    ├── gpt-4.ipynb                                <- Script for running GPT-4 model.
    └── run_open_llms.ipynb                        <- Script to run open-source LLMs.


## Contact


## Find a bug?
If you found an issue or would like to submit an improvement to this project, please contact the authors. 

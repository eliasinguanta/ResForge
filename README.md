# ResForge

A research project comparing different combinations of incremental learning and transfer learning.

## Overview

ResForge is a machine learning research project that investigates the impact of transfer learning techniques such as fine tuning and feature extraction on incremental learning methods like distillation and representative memory. The CIFAR 100 dataset is used, with 10 new classes added in each of 5 steps, continuing training after each increment. The experiment is implemented in Python using PyTorch and executed on AWS SageMaker using one ml.g4dn.xlarge instance.

## Features

- Implementation of distillation and representative memory
- Integration of transfer learning techniques such as fine tuning and feature extraction by freezing between 0 and 6 layers (ranging from full fine tuning to pure feature extraction)
- AWS SageMaker integration for training

## Prerequisites

- AWS Account with SageMaker access
- Python 3.10+
- Required Python packages (see `requirements.txt`)

## Setup 

1. Clone the repository:
```bash
git clone https://github.com/eliasinguanta/ResForge
cd ResForge
```

2. Create and activate virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure AWS credentials:
```bash
aws configure
```

5. Create the cloud infrastructure as described in sagemaker/terraform/README.md

## Usage

Run the current experiment:
```bash
python3 sagemaker/run.py \
  --role arn:aws:iam::<account-id>:role/<training-role> \
  --bucket <s3-bucket-name>
```

Show the results of the experiment:
```bash
python3 visualize/show.py \
  --prefix models/pytorch-training-2025-06-23-05-52-26-008/output \ 
  --bucket <s3-bucket-name> \
  --local_dir ./tensorboard_logs
```
The prefix is just an example here.


## Project Structure

```
ResForge/
├── sagemaker/         # SageMaker setup
├── src/               # Source code (could be run without sagemaker)
└── visualize/         # Tensorboard code
```

## License

MIT

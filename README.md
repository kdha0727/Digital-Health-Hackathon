# [Digital Health Hackathon 2021](https://www.digitalhealthhack.org)

**Source Code**

> Team Information
> - Team 삼김
> 
> Participants
> - **[Dongha Kim](https://github.com/kdha0727)**
> - **[Jihwan Kim](https://github.com/gitubjimmy)**
> - **[Jinyeong Kim](https://github.com/rubato-yeong)**
> - **[Seungwoo Cho](https://github.com/brandoncho321)**
> 
> Git Origin
> - [gitubjimmy/digital-health-hackathon](https://github.com/gitubjimmy/digital-health-hackathon)
>

# Setup

* Setup python environment
```bash
$ pyenv install 3.7.10
$ pyenv virtualenv 3.7.10 digital-health-hackathon
$ pyenv activate digital-health-hackathon
$ pip install -r requirements.txt
```

# Methods

## EDA

```bash
$ python eda.py
```

## Cox Regression

* Main(Jupyter)
  * [cox_regression.md](./cox_regression.md)

## Bayesian Network

* Preprocessing (Python)
  * [bayesian_preprocess.py](./bayesian_preprocess.py)
* Main (R)
  * [bayesian_process.R](./bayesian_process.R)

## MLP

```bash
$ python mlp_train.py
$ python mlp_test.py
```

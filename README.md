# FashionMNIST classification task
This repository shows a solution to the classification problem of MNIST / FashionMNIST using Pytorch & Deep Learning.

This exercice is from a CentraleSupélec course: 

https://github.com/jeremyfix/deeplearning-lectures/tree/master/LabsSolutions/00-pytorch-FashionMNIST

## How to setup

To create & start the virtual environment:
```Shell
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Or, to load an already existing venv:
```Shell
source venv/bin/activate
```

```Shell
sh11:~:mylogin$ tensorboard --logdir ./logs
Starting TensorBoard b'47' at http://0.0.0.0:6006
```

## How to run

The arguments of a run (which model & which dataset) have to modified in the **main.py** file

```Python
    main(
        dataset_src=DATASET.MNIST,
        model_type=MODEL_TYPE.VANILLA_CNN,
    )
```
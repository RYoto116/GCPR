# GCPR
This is the PyTorch implementation for our model GCPR.

This project is based on [SGL-Torch](). Thanks to the authors.

## Environment Requirement

The code runs well under python x.x.x. The required packages are as follows:

- pytorch == 1.9.1
- numpy == 1.20.3
- scipy == 1.7.1
- pandas == 1.3.4
- cython == 0.29.24

## Quick Start
**Firstly**, compline the evaluator of cpp implementation with the following command line:

```bash
python setup.py build_ext --inplace
```

If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.

**Note that the cpp implementation is much faster than python.**

**Secondly**, change the value of variable *root_dir* and *data_dir* in *main.py*, then specify dataset and recommender in configuration file *NeuRec.ini*.

Model specific hyperparameters are in configuration file *./conf/GCPR.ini*.

Some important hyperparameters:

### Movielens-10M dataset
```
aug_type=ED
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.1
ssl_ratio=0.1
ssl_temp=0.2
```

### amazon-book dataset
```
aug_type=ED
reg=1e-4
embed_size=64
n_layers=3
ssl_reg=0.5
ssl_ratio=0.1
ssl_temp=0.2
```

### ifashion dataset
```
aug_type=ED
reg=1e-3
embed_size=64
n_layers=3
ssl_reg=0.02
ssl_ratio=0.4
ssl_temp=0.5
```

**Finally**, run [main.py](./main.py) in IDE or with command line:

### yelp2018 dataset
```bash
python main.py --recommender=SGL --dataset=yelp2018 --aug_type=ED --reg=1e-4 --n_layers=3 --ssl_reg=0.1 --ssl_ratio=0.1 --ssl_temp=0.2
```

### amazon-book dataset
```bash
python main.py --recommender=SGL --dataset=amazon-book --aug_type=ED --reg=1e-4 --n_layers=3 --ssl_reg=0.5 --ssl_ratio=0.1 --ssl_temp=0.2
```

### ifashion dataset
```bash
python main.py --recommender=SGL --dataset=ifashion --aug_type=ED --reg=1e-3 --n_layers=3 --ssl_reg=0.02 --ssl_ratio=0.4 --ssl_temp=0.5
```

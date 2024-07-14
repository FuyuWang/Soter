# Soter

### Setup ###
* Download the Soter source code 

* Create virtual environment through anaconda
```
conda create --name SoterEnv python=3.8
conda activate SoterEnv
```
* Install packages
   
```
cd Soter
pip install -r requirements.txt
```

* Install [Timeloop](https://timeloop.csail.mit.edu/timeloop)

### Run Soter on Simba ###

```
cd Simba
CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj latency --epochs 10 --accelerator Simba --workload mobilenet --layer_id 1 --batch_size 1
```


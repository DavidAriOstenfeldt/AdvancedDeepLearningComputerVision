# nerf-project

## Installation
* `conda create -n vfi python=3.10.10`
* `conda activate vfi`
* `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
* `pip install -r requirements.txt`
* `python download_davis.py`
* `python extract_triplets.py`

## Train and evaluate your model
We define `conf-num` as the index of the loss function to use (1-3)
* `python train.py --conf <conf-num>`
* `python slow_movie.py --conf <conf-num>`


## Logging
* The code uses tensorboard to log the train loss. Use the command `tensorboard --logdir=runs` to observe the training loss.
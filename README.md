Splendor AI [![PyPI](https://img.shields.io/pypi/pyversions/Django.svg?style=plastic)](https://github.com/filipmlynarski/splendor-ai)
===========
This project contain code to train a model to play board game Splendor.

Installation
------------
```
git clone https://github.com/filipmlynarski/splendor-ai.git
sudo apt-get install python3-tk
pip3 install -r requirements.txt
```

Train model
-----------
```
cd splendor_ai
python3 train_model.py model_name
```

Try Playing Yourself
-----------
```
python3 interactive_splendor.py p model_name
```
![alt text](https://raw.githubusercontent.com/filipmlynarski/splendor-ai/master/interactive_splendor.png)
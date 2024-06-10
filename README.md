# NTMGraph
The data and code of paper Topic Aware Graph

# 1. Preprocess the data Gossipcop
The format of dataset of the model NTM we used is in `ntm/data/gossipcop_lines.txt`, which you can find that each line in the txt file is one doc of your datas.

So first to preprocess to get the txt file:

```bash
$ python preprocess.py
```
And in the `preprocess.py` you can change the data path. 

The dataset `gossipcop_v3_keep_data_in_proper_length.json` we offered here has 15,729 news. The 0-3,784 are fake news, and the other 3785-1,5729 are real news. Detailes can be seen in [CossipCop-LLM](https://github.com/SZULLM/GossipCop-LLM).

# 2. Run the NTM model
```bash
cd ntm
python GSM_run.py --taskname gossipcop
```
Then you will find the model ckpt in `ckpt/`, and change the ckpt path in the next code `embedding.py`.
```bash
python embedding.py
```
After that you could get the docs embedding under topic in `results/gossipcop_embed.py`, whose size is (n, 16).(The n_topic we set is 16).

The docs of different topic is stored in `results/gossipcop_topic_{i}.txt`, where i is from 0 to 15.

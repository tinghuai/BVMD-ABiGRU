# BVMD-Attention-BiGRU 
Code for the paper “Mid-Long Term Drought Prediction Based on BVMD-Attention-GRU”.

* Make sure the following files are present as per the directory structure before running the code：
```
├── data
|     ├── all-data (Meteorological data used in the experiment)
|     │    ├── Urumqi1951-2020-data.csv
|     │    ├── hanzhong1951-2020-data.csv
|     │    ├── yanan1951-2020-data.csv
|     │    ├── yulin1951-2020-data.csv
├── models
|    ├── bert-base-chinese
|    |     ├── config.json
|    |     ├── pytorch_model.bin
|    |     └── vocab.txt
|    ├── bert-base-uncased
|    |     ├── config.json
|    |     ├── pytorch_model.bin
|    |     └── vocab.txt
|    ├── config.py
|    ├── data.py
|    ├── data_process.py
|    ├── layers.py
|    ├── main.py
|    ├── models.py
|    ├── path_zh.py
|    └── util.py   
├── model_saved
├── preprocess
|    ├── getTextEmbedding.py
|    ├── getTwittergraph.py
|    ├── getWeibograph.py
|    ├── pheme_pre.py
|    ├── stop_words.txt
|    └── weibo_pre.py
└── requirement.txt
```

## Dependencies
* torch_geometric==2.5.2
* torch_scatter==2.1.2
* torch==1.12.1
* scipy==1.5.4
* tqdm==4.63.1
* numpy==1.21.5
* pandas==1.1.5
* visdom==0.2.4
* transformers==4.17.0
* jieba==0.42.1


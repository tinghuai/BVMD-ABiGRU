# BVMD-Attention-BiGRU 
Code for the paper “Mid-Long Term Drought Prediction Based on BVMD-Attention-GRU”.

* Make sure the following files are present as per the directory structure before running the code：
```
├── BVMD-ABiGRU_Project
|     └── data
|          ├── all-data (Meteorological data used in the experiment)
|          │    ├── Urumqi1951-2020-data.csv
|          │    ├── hanzhong1951-2020-data.csv
|          │    ├── yanan1951-2020-data.csv
|          │    ├── yulin1951-2020-data.csv
|     └── SPEI
|          ├── program (SPEI calculation program)
|          │    ├── spei.exe
|          │    ├── spei_manual_en.pdf
|     └── SPI
|          ├── program (SPI calculation program)
|          │    ├── DocumentFormat.OpenXml.dll
|          │    ├── DocumentFormat.OpenXml.xml
|          │    ├── SPIGenerator.exe
|          │    ├── SPIGenerator.exe.config
|          │    ├── SPIGenerator.pdb
|          │    ├── SPIGenerator.xml
|          │    ├── StandardPrecipitationIndex.dll
|          │    ├── StandardPrecipitationIndex.pdb
|     ├── bigru_attention.py
|     ├── data_process.py
|     ├── DWT.py
|     ├── EEMD.py
|     ├── main.py
|     ├── mic.py
|     ├── SampEntropy.py
|     ├── util.py
|     ├── vmdpy.py
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


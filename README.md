Segmentation of data contests
========

The company where I work runs online machine learning contests. This python code performs an unsupervised clustering analysis using the metadata of these challenges, along with other measurements of user behavior over their run.

Author: RJ Ramey

(c) 2014 by RJ Ramey. All rights reserved. 

No license is granted at this time. Stay tuned.

### Inputs

This code contains a series of functions that operate on inputs in location /data/raw/ :

- csv files from company's internal SQL query results  (These do not appear in this repository)

### data_cleaner.py
- reads csvs and encodes feature names
- enforces data types: converts boolean to binary 0/1, timestamps as strings to datetime64
- drops unnecessary features
- **Output :** clean and coded *filename*.csv in /data/cleaned/

### feature_engineer.py
- creates 80 derived measurements from tables of cleaned data
- assembles outputs: JOIN tables on Id
- **Output :** M*xx*.csv with engineered features, indexed by Id in /data/cleaned/
- **Output :** /data/cleaned/Final.csv containing all calculated features

### feature_visualizers.py
- **Output :** histogram plots to /data/vis/ 

### run_cluster.py
- assesses whether to fill NaN or drop rows
- scales each feature to mean 0, std 1 in (-1, 1)
- performs KMeans clustering
- evaluates iteratively among value of k to maximize the Silhouette score
- **Output :** 
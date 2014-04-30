Segmentation of data contests
========

The company where I work runs online machine learning contests. This python code performs an unsupervised clustering analysis using the metadata of these challenges along with other measurements of user behavior over their run.

Author: RJ Ramey

(c) 2014 by RJ Ramey. All rights reserved. 

No license is granted at this time. Stay tuned.

### Inputs

This code contains a series of functions that operate on inputs in location /data/raw/ :

- csv files from company's internal SQL query results  (These do not appear in this repository)

### data_cleaner()
- reads csvs and encodes feature names
- enforces data types: converts int to float, boolean to binary, times as strings to datetime64
- drops unnecessary features
- **Output :** clean and coded *filename*.csv in /data/cleaned/

### feature_engineer()
- creates 20-45 derived measurements from tables of cleaned data
- **Output :** *filename*_measure.csv with calculated measurements indexed by Id in /data/cleaned/

### feature_visualizer()
- **Output :** histogram plots to /data/vis/ 

### data_normalizer()
- **Output :** *filename_measure*_norm.csv for measurements csvs in /data/cleaned/

### run_cluster()
- assembles outputs: JOIN tables on Id to each *filename_measure*_norm.csv
- performs KMeans clustering
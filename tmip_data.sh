#!/bin/bash
echo Downloading Chest X-Ray Data from COVID-19 image data collection
git clone https://github.com/ieee8023/covid-chestxray-dataset.git
pip install gdown
echo Downloading Kaggle RSNA Pneumonia Detection Challenge Data 
gdown https://drive.google.com/uc?id=1fjAJO5ruSOMIjWTQAnDxOcQs7NZd5bIv
echo Unzip Kaggle Data to  RSNA-Data Folder
unzip rsna-pneumonia-detection-challenge.zip -d rsna-data

#!/bin/bash
echo Preprocessing Chest X-Ray Data
mkdir data/processed/
mkdir data/processed/train/
mkdir data/processed/test/
mkdir data/processed/val/
mkdir data/interpretability/
python ./src/data/preprocess.py


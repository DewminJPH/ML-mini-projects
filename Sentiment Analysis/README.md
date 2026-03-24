# Sentiment Analysis Project

This project implements a sentiment analysis model to classify tweets as positive or negative.

## Dataset

The dataset used is the "Sentiment Analysis Dataset" from Kaggle (by dineshpiyasamara), containing tweets labeled for sentiment analysis.

- **Source**: [Kaggle Dataset](https://www.kaggle.com/dineshpiyasamara/sentiment-analysis-dataset)
- **Labels**: 0 (positive), 1 (negative)

## Project Structure

- `notebooks/`: Jupyter notebooks for data processing and model building
  - `download_dataset.ipynb`: Downloads and extracts the dataset from Kaggle
  - `model_building.ipynb`: Data preprocessing, feature engineering, and model training
- `artifacts/`: Dataset files and Kaggle API credentials
- `static/model/`: Model artifacts (vocabulary, corpora)
- `env/`: Python virtual environment

## Setup Instructions

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   env\Scripts\activate  
# Facial Emotion Recognition System

## Abstract
This project implements a simple **Facial Emotion Recognition (FER) system** using **ResNet34** with **fine-tuning** on the **FER2013 dataset**. 
The system can classify facial expressions into multiple emotion categories, providing a baseline for further research or deployment in applications such as human-computer interaction.

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/nhut-nam/FaceEmotionRecognitionSystemTest.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n FER python=3.10 -y
```

```bash
conda activate FER
```

### STEP 02- Install requirements and run setup
```bash
cd FaceEmotionRecognitionSystem
pip install -r requirements.txt
pip install -e .
```
Optional: install MLflow if you want experiment tracking
```bash
pip install mlflow
```

### STEP 03- TRAIN
You can run:
```bash
python main.py
```
or use DVC pipeline:
```bash
dvc repro
```

### STEP 04- RUN APP
```bash
python app.py
```
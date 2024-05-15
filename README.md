# Projects in Data Science - Medical Imaging - group R

## Purpose
The aim of the project is to build a model for predicting if a lesion is cancerous or not.

## How to use
* To run "01_process_images.py" set path to images, masks, and metadata. The script will create a csv file of the features needed for training.

* To run "02_train_classifiers.py" set path for training data. The script 
will create most optimized models using the training data.

* To run "03_evaluate_calssifier.py" use classify function as an evaluator for a single image. This script will classify the lesion.

# Evaluation of Task-fMRI Decoding with Deep Learning on a Small Sample Sized Dataset

This is a repository of codes for Evaluation of Task-fMRI Decoding with Deep Learning on a Small Sample Sized Dataset.

### requirements
- scikit-learn == 0.22.2.post1
- pytorch == 1.3.1
- captum == 0.2.0
- nipy == 0.4.2
- nilearn == 0.6.2

## 1. Prapare dataset
We should set `config.ini` for data directory, fMRI image file name, label file name, and subjects id.

`python scripts/prepare_dataset.py`


## 2. Train classifiers
We can select three split methods and four train methods. 

`python scripts/train.py [split_method] [train_method]`

|args|options|
|--|--|
|split method|LeaveOneSubjectOut, SessionShuffleSplit, SampleShuffleSplit|
|train method|PLR, SVM, M2DCNN, 3DCNN|


## 3. Analyze results
### 3.1 Permutation test
We can calculate the chance level for each split methods.

`python scripts/permutation.py [split_method] [permute_number]`


### 3.2 Calculate Integrated Gradients
We can calculate Integrated Gradients for DL model.

`python scripts/BackProp.py`

### 3.3 Aggregate and Visualize results

`python scripts/visualize.py`  
`python scripts/aggregate.py`  




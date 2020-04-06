# Machine Learning Engineer Nanodegree, Capstone project
To reproduce the code, please read the following remarks. 

### Overview
- **01_data_exploration.ipynb:**: Data exploration tasks such as loading the data,  visual inspection of the dataset and calculating some simple statistics. (Kernel used: conda_python3)

- **02_benchmark_model.ipynb:** Implementing two benchmark models to measure the final Deep Learning model against. (Kernel used: conda_python3)

- **03_keras_model.ipynb:** Running Sagemaker training and deploy jobs as well as evaluating the final results and create a submission file for kaggle. (Kernel used: conda_tensorflow_p36)

In **./src/**  the code for the Deep Learning algorithm as well as other helper code for visualizations is stored. You don't need to install additional libraries, the kernels mentioned have them pre-installed already. 

### Prerequisites
The reproduce the results, you need to create the following file structure: 
- ./data: Here you need to store the train.csv, test.csv and submission.csv. You find those files here: [Kaggle](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data)
- ./images: All the train and test images from [Kaggle](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data) need to be stored here.

If those folders and files are available, you should be good to go. 


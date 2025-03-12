# Repository Structure
````

├── Data
│   ├── TASK-ML-INTERN.csv
├── models
│   ├── model_{current_time}.pkl
├── notebooks
│   ├── ImagoAI_task_01.ipynb
├── train.py
├── visualize.py
├── requirements.txt
├── Task_report.pdf
└── README.md

````

- Data Folder consists of the dataset in .csv format
- models directory is where the saved models are stored
- notebooks is the directory to store all experimentation notebooks
- train.py is the main file for training model
- visualize.py is the file for visualizing the data before training 
- preprocess.py is the file which contains all prepocess steps (i.e handling missing values and applying standardScaler)
- It also contains the task report which explain a brief about the approach to the solution.
# How to Run the model
- Open terminal and enter the following command
````
git clone https://github.com/dwij-patel123/ImagoAI_Task.git
````
- To install depedencies run the following command
````
pip install -r requirements.txt
````
- To visualize the data and get insights about data run the following command
````
python visualize.py
````
- To train the model(Model used is CatboostRegressor) run the following command
````
python train.py
````
- The model will be saved in the ```models``` directory in ```.pkl``` format which can be used for inference any time later
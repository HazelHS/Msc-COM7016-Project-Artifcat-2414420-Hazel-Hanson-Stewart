Currently, this project requires jupyter notebook support. It is recommended to use visual studio code, with the appropriate extentions installed.

1. Run the command "pip install -r requirements.txt" from the root directory of the repo, after creating a fresh environment.

2. Then run Dataset_Collector.py, this will create an apporpriate .csv file with the compiled features required for the base model. 

3. Next, open Dataset_Processor.ipynb and run each code block in succession, this is the processing steps for the base model's dataset, each block will explain its function and purpose.

- the training of the models locks in the target feature, which is key to how hyperparameter tuning works.

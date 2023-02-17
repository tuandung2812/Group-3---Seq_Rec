- To run the experiments in this project, run all the bash files in the src folders
- src: Directory to hold the source code for the dataloaders, models, data augmentation and loss functions
    + helpers:
        + BaseReader.py, SeqReader.py: src code to load the data
        + BaseRunner.py: API to train, save, evaluate our models and methods
    + models:
        + BaseModel.py: Abstract class to implement the specific models
            + Contains methods to process and augment data, save and load models, fit different loss functions for training
    + utils: different helper functions
    + main.py: main file for running the project 
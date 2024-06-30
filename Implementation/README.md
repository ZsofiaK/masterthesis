# Implementation of the project

This directory contains the implementation of all the models and experiments used in the thesis.

**PLEASE NOTE THE FOLLOWING:**

* Most of the project is implemented in the form of Jupyter notebooks. Each notebook starts with a cell (or cells) in which the parameters to run the notebook (e.g., dataset to use, directories to read from / save to) can be set.

* The terminology used in the notebooks can sometimes differ slightly from the one used in the thesis paper. Most notably, the coral reef dataset is referred to as `fishClips` in the code, the AK fish dataset as `AK-fish`, and the representative sample of Animal Kingdom used in the multi-label setting as `AK-sample`.

* The notebooks rely on the datasets being stored on Google Drive. Notebooks in the _Auxilliary_ folder ensure that the datasets are set up in the correct manner. However, all notebooks can of course be adapted to read the data and save results to other locations.

---
The directory is organized in the following folders.

## MARINE
Contains the implementation of the MARINE model, including all experiments run to apply it to the coral reef and AK fish datasets.

A README file in this folder clarifies how the model can be run.

## Auxilliary
Contains notebooks which help set up the datasets used in the project and run process checks to ensure that they were set up correctly. The notebooks setting up the datasets and creating the train-test splits (indicated by title) should be run before executing the MARINE model to ensure that the datasets are in the correct format.

Furthermore, this directory includes two Python modules, `frame_selection` and `video_processing`, which are imported and used in some of the notebooks in the _Implementation_ directory. In these notebooks, the location of the module files are specified in the starting cells, and should be changed according to where the project is run (e.g., in Colab using Google Drive, or locally from the cloned repository).

## Benchmarking
Contains notebooks which run the benchmarking experiments for the thesis. Specifically, applying VideoMAE to the AR task and MARINE G14 to the representative sample in the multi-label setting can be executed here.

## EDA
Contains notebooks related to Exploratory Data Analysis on the datasets, most importantly AK fish.

## assets
Contains metadata on the Animal Kingdom and AK fish datasets.

## Experimentation
Contains notebooks which are not directly relevant to executing the experiments for the thesis, but instead were used to explore different techniques and datasets, some of which were eventually integrated into the implementation of the project.

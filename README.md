# GAIL (Geisinger AI Lab) L3C Submission

This is the public repository for the Geisinger AI Lab (GAIL) submission to the [NIH Long COVID Computational Challenge (L3C)](https://www.challenge.gov/?challenge=l3c)

Our submission received [second place overall](https://www.linkedin.com/pulse/announcing-nih-long-covid-computational/)! Congratulations to the other finalists and honorable mentions, Convalesco (U Chicago / Argonne National Lab), UC Berkeley, UW Madison, U Penn, and Ruvos. 

This repository includes all of the code we submitted as a part of the L3C. Initially developed within the [N3C Enclave](https://covid.cd2h.org/enclave), the code is adapted here to run locally with csv files in the [OMOP Common Data Model](https://ohdsi.github.io/CommonDataModel/) format. A small sample of simulated claims data (synpuf) is included in the repo for demo purposes. 

## Directory structure

- `src/`
    - `src/features/`: python functions to read data and output formatted feature sets, organized by feature type (demographics, labs, meds, etc.)
    - `src/training/`: python functions to train the model
    - `src/inference/`: python functions run the trained model on the test set
    - `src/pipeline/`: python scripts to 1) process raw data and save a table of features 2) train the model 3) run inference on the test set
- `documents/`: The GAIL L3C submission write-up
- `data/`: raw OMOP synpuf data, as well as interum data files created during featurization, the featurized data sets, and predictions
- `models/`: to write models created during training, and read trained models for inference
- `notebooks/`: supplementary code notebooks 

## Getting started

Common tasks are supported by the Makefile at the project root. A linux environment needs build-essential installed in order to use make, e.g. `sudo apt-get install build-essential` on Ubuntu. Also, pyspark requires Java, e.g. `sudo apt-get install default-jre` on Ubuntu. The data science node has these installed already. I haven't tried on Windows directly, so let me know if you run into issues. Also, if you prefer using conda instead of venv for python virtual environments, please feel free to use the tool you like.

After cloning this repo and cd'ing into the project root, create a virtual environment using the Makefile
```sh
make venv
```

You will be propted to activate the environment. Run the following to do so:
```sh
source .venv/bin/activate
```

Now you can install requirements via pip, as well as the local source. 
```sh
make requirements
```

After making changes, you can format the code before committing.
```sh
make format
```

The project is structured with high level pipeline steps in `src/pipeline`. All configuration for the project can be placed in `params.yaml` at the project root. All pipeline steps, e.g. featurize, take the config filepath as an argument. The pipeline steps can be run as follows:

Featurize:
```sh
python src/pipeline/featurize.py --config params.yaml
```

Train:
```sh
python src/pipeline/train.py --config params.yaml
```

Infer:
```sh
python src/pipeline/infer.py --config params.yaml
```

Pre-configured param files are created for the demo training and testing data samples, which can be run using make commands

Run featurize, training, and inference for the training set:
```sh
make run-training
```

Run featurize inference for the testing set:
```sh
make run-testing
```


    
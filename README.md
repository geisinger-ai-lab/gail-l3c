# GAIL (Geisinger AI Lab) L3C Submission

## Who's working on what?

| Feature type   | OMOP table  | owner |
| -------------- | ----------- | ----- |
| Demographics   | person      | Gaurav
| Lab results    | measurement | Gaurav/Tamanna
| Vitals         | measurement | Elliot
| Diagnoses      | contition   | Gaurav/Biplab
| Procedures     | procedure   | Tamanna
| Utilization    | visit       | Grant
| Medications    | drug        | Elliot
| Smoking Status | observation | Elliot
| Index Range    | micro/macro | Grant

Model train/test: Gaurav

makefile/venv/global_code/setup.py: Grant 

## Directory structure

- `raw/`: raw export from the N3C enclave workbook
- `src/`
    - `src/features/`: python functions to read raw data and output formatted feature sets, organized by feature type (demographics, labs, meds, etc.)
    - `src/training/`: python functions to train the model
    - `src/inference/`: python functions run the trained model on the test set
    - `src/pipeline/`: python scripts to 1) process raw data and save a table of features 2) train the model 3) run inference on the test set
- `documents/`: the L3C write-up
- `data/`: home for raw train/test OMOP synpuf data, as well as interum data representations created during data preparation

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

The individual transforms originally created in the enclave can be mapped to lower level funcitons within `src`. For example, the utilization transforms would be mapped to functions in `src/features/utilization.py`. The utilization module also has a 'main' function called `get_utilization`, which would call all of the individual transform functions and return the resulting utilization DataFrame. This way, the featurize pipeline step can make one call to each feature module and then merge the DataFrames as needed for the model. 

As of this writing, the data is not available in the repo. Once available, we can use `spark.read.csv` to read the raw data. I suggest all subsequet read/write operations should use parquet format for efficiency, but I'm open to other suggestions.

In general, when you are ready to make changes, do a `git pull` first to get the latest version of the repo. Then create a new branch with `git checkout -b {my_new_branch}` so that your changes go into the new branch instead of directly into main. After pushing your work to the remote repo on the new branch, create a pull request on GitHub whenever you'd like others to review or you want to merge into main. You can look at my [git guide](https://github.com/g-delong/git_guide) for more info on using git, but of course there's a ton of other info out there on the webs. 




    
# GAIL (Geisinger AI Lab) L3C Submission

## Who's working on what?

| Feature type   | OMOP table  | owner |
| -------------- | ----------- | ----- |
| Demographics   | person      | Gaurav
| Lab results    | measurement | Tamanna
| Vitals         | measurement | Elliot
| Diagnoses      | contition   | Biplab
| Procedures     | procedure   | Tamanna
| Utilization    | visit       | Grant
| Medications    | drug        | Elliot
| Smoking Status |observation) | Elliot

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

    
# Scripts to generate Patient Context Vectors and related ML models

```
@article{apostolova2019combining,
  title={Combining Structured and Free-text Electronic Medical Record Data for Real-time Clinical Decision Support},
  author={Apostolova, Emilia and Wang, Tony and Koutroulis, Ioannis and Tschampel, Tim and Velez, Tom},
  journal={BioNLP 2019},
  year={2019}
}
```

## Prerequisites

### MIMIC 3 Postgres Database

### Python 3.6 and pip

### Tensforflow and Keras

### Project dependencies

`pip install -r requirements.txt`

## Scripts


### Generate ICD embeddings from all MIMIC data

Set the following env variables


```bash
    export POSTGRES_USER="username"
    export POSTGRES_PASSWORD="password"
    export POSTGRES_DATABASE="mimic"
    export ICD_EMBEDDINGS_FILE="/tmp/icdembeddings.mimic.50.bin"

```

Execute:

`python icd_embeddings.py`

The script generates embeddings of size 50 and saves it to $ICD_EMBEDDINGS_FILE.


### Cluster ARDS patients based on ICD embeddings


Set the following env variables

```bash
    export POSTGRES_USER="username"
    export POSTGRES_PASSWORD="password"
    export POSTGRES_DATABASE="mimic"
    export ICD_EMBEDDINGS_FILE="/tmp/icdembeddings.mimic.50.bin"
    export NUM_CLUSTERS=10
    export EMBEDDING_SIZE=50
```

The script uses a new column in the mimic db to store cluster values for faster lookup:

`
alter table mimiciii.diagnoses_icd add column cluster int default -1;
`

Execute:

`python ards_clusters.py`


### Script to generate an ARDS clusters chart

Execute:

`python charts.py`


### Script to train a network to predict Patient Context Vectors from notes

Set the following env variables

```bash
    #input files 
    export TRAINING_DATA_FILE="nurse_phys_mimic_notes_w_hadm_id_50.csv"
    export DICT_FILE="all_mimic_notes.dict"
    export EMBEDDINGS_MODEL_FILE="all_mimic_notes.txt.100.7.bin"
    
    #generated from the previoius script
    export ICD_EMBEDDGINS_FILE="mimic_icd_shuffled.txt.50.bin"
    
    #ouptut
    export CNN_MODEL_FILE="/tmp/cnn.model"
```


Execute:

`python notes2pcv.py`


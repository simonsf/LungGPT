# LungGPT

 Expert-level large language model for tailored respiratory disease diagnosis and treatment

##  Data
Due to the sensitivity of clinical data, it will not be made public. 
Desensitized cases are provided for reference instead.
 
## Data Preparation
### Modify your data format referring to the test case in ./data/data.json before your training or predicting.
- input: the patient's chief complaint, present medical history, past medical history and Other information beneficial for doctors' diagnosis.
- output: doctor's diagnosis Groud truth.

## Run
### Modified json config file in  ./config/hyperparameters.json before running the following predicting code.
`python ./example/api.py`









































































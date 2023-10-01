# CSC722 Persona and Knowledge Empowered Conversational Model

Source codes for CSC722 course project "Persona and Knowledge Empowered Conversational Model". 

This implementation is based on the paper **[Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge](https://arxiv.org/abs/2112.08619)**
and its **[source code](https://github.com/pkchat-focus/FoCus)**.


## Environment Setup
We trained the models under the setting of `python==3.8` and `torch==2.0.1`.

1.Make a virtual environment
```    
conda create -n ENV_NAME python=3.8
```

2.Activate virtual environment
```
conda activate ENV_NAME
```   


3.Install `cuda-toolkit`
```
conda install --channel "nvidia/label/cuda-11.8.0" cuda-toolkit
```

4.Install `torch`
```
pip3 install torch torchvision torchaudio
```

5.Install the required libraries.
```
pip install -r requirements.txt
```
    
## Downloading Dataset

Please refer to the `Dataset` section below from the original [FoCus Dataset paper](https://arxiv.org/abs/2112.08619)
to download the dataset and put into correct file path. 

## Run Program

The script `csc722_runscripts.sh` includes the commands that we run to train/validate the model. Below are some examples.

### Training
**Train with GPT-2 model**
```commandline
python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_FOCUS --kp_method focus --model_name GPT2 --incontext > train_log_focus_e2_L10_gpt.log 2>&1
```
**Train with BART model**
```commandline
python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_FOCUS --kp_method focus --model_name BART --incontext > train_log_focus_e2_L10_bart.log 2>&1
```

### Testing
**Test with GPT-2 model**
```commandline
# Generation metrics
python evaluate_test.py --kp_method focus --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E2_L10_FOCUS > test_log_focus_txt_test_GPT_MTL.log 2>&1
# Perplexity
python evaluate_test_ppl.py --kp_method focus --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E2_L10_FOCUS > test_log_focus_txt_test_GPT_MTL_ppl.log 2>&1
```
**Test with BART model**
```commandline
# Generation metrics
python evaluate_test.py --kp_method focus --model_name BART --model_checkpoint models/train_focus_BART_E2_L10_FOCUS > test_log_focus_txt_test_BART_MTL.log 2>&1
# Perplexity
python evaluate_test_ppl.py --kp_method focus --model_name BART --model_checkpoint models/train_focus_BART_E2_L10_FOCUS > test_log_focus_txt_test_BART_MTL_ppl.log 2>&1
```


## Additional Information from original [FoCus Dataset paper](https://arxiv.org/abs/2112.08619)

### Dataset [**[FoCus dataset v2](https://drive.google.com/file/d/1YmEW12HqjAjlEfZ05g8VLRux8kyUjdcI/view?usp=sharing)**]
This data is the modified version of the original data (which is reported in the paper) after ethical inspection.

| FoCus v2 STATISTICS | Train | Valid |
| --- | --- | --- |
| `# dialogues` | 12,484 | 1,000 |
| `# avg rounds` | 5.63 | 5.64 |
| `# knowledge-only answers` | 37,488 | 3,007 |
| `# persona-knowledge answers` | 32,855 | 2,630 |
| `# landmarks` | 5,152 | 923 |
| `avg len of Human's utterances` | 40.70 | 40.21 |
| `avg len of Machine's utterances` | 138.16 | 138.60 |

You should create directories named **`infer_log_focus`, `train_log_focus`, `test_log_focus`, `models`, `data`** under FoCus folder.

We put train, valid, test files of the dataset in the **`data`** folder. (The test set will be available after March 2022.)

The project directory should follow this directory structure:


    📦FoCus
    ┣ 📂data
    ┃ ┗ 📜train.json
    ┃ ┗ 📜valid.json
    ┣ 📂ignite
    ┣ 📂infer_log_focus
    ┣ 📂models
    ┣ 📂python_tf_idf
    ┣ 📂test_log_focus
    ┣ 📂train_log_focus
    ┣ 📜classification_modules.py
    ┣ 📜data_utils.py
    ┣ 📜evaluate_test.py
    ┣ 📜evaluate_test_ppl.py
    ┣ 📜inference.sh
    ┣ 📜inference_test.py
    ┣ 📜LICENSE
    ┣ 📜README.md
    ┣ 📜requirements.txt
    ┣ 📜test.sh
    ┣ 📜train.sh
    ┣ 📜train_focus.py
    ┗ 📜utils_focus

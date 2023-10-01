############################
# On tiny dataset
############################
# train GPT2/BART
# focus
python train_focus.py --n_epochs 1 --lm_coef 10 --flag E1_L10_FOCUS --kp_method focus --model_name GPT2 --incontext --train_dataset_path data/train_focus_tiny.json --dev_dataset_path data/valid_focus_tiny.json
python train_focus.py --n_epochs 1 --lm_coef 10 --flag E1_L10_FOCUS --kp_method focus --model_name BART --incontext --train_dataset_path data/train_focus_tiny.json --dev_dataset_path data/valid_focus_tiny.json
# cross attention
python train_focus.py --n_epochs 1 --lm_coef 10 --flag E1_L10_CROSSATTN --kp_method cross_attention --model_name GPT2 --incontext --train_dataset_path data/train_focus_tiny.json --dev_dataset_path data/valid_focus_tiny.json
python train_focus.py --n_epochs 1 --lm_coef 10 --flag E1_L10_CROSSATTN --kp_method cross_attention --model_name BART --incontext --train_dataset_path data/train_focus_tiny.json --dev_dataset_path data/valid_focus_tiny.json

# test GPT2/BART
# focus
python evaluate_test.py --kp_method focus --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E1_L10_FOCUS --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
python evaluate_test_ppl.py --kp_method focus --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E1_L10_FOCUS --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
python evaluate_test.py --kp_method focus --model_name BART --model_checkpoint models/train_focus_BART_E1_L10_FOCUS --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
python evaluate_test_ppl.py --kp_method focus --model_name BART --model_checkpoint models/train_focus_BART_E1_L10_FOCUS --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
# cross attention
python evaluate_test.py --kp_method cross_attention --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E1_L10_CROSSATTN --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
python evaluate_test_ppl.py --kp_method cross_attention --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E1_L10_CROSSATTN --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
python evaluate_test.py --kp_method cross_attention --model_name BART --model_checkpoint models/train_focus_BART_E1_L10_CROSSATTN --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz
python evaluate_test_ppl.py --kp_method cross_attention --model_name BART --model_checkpoint models/train_focus_BART_E1_L10_CROSSATTN --test_dataset_path data/valid_focus_tiny.json --test_dataset_cache data/focus_cache.tar.gz

# after done, remove all the caches
rm -r models/*E1_L10/
rm -r data/focus_cache.tar.gz*

############################
# On full dataset - FoCus
############################
# train GPT2/BART
python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_FOCUS --kp_method focus --model_name GPT2 --incontext > train_log_focus_e2_L10_gpt.log 2>&1
python train_focus.py --n_epochs 2 --lm_coef 10 --flag E2_L10_FOCUS --kp_method focus --model_name BART --incontext > train_log_focus_e2_L10_bart.log 2>&1


# test GPT2/BART
python evaluate_test.py --kp_method focus --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E2_L10_FOCUS > test_log_focus_txt_test_GPT_MTL.log 2>&1
python evaluate_test_ppl.py --kp_method focus --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E2_L10_FOCUS > test_log_focus_txt_test_GPT_MTL_ppl.log 2>&1
python evaluate_test.py --kp_method focus --model_name BART --model_checkpoint models/train_focus_BART_E2_L10_FOCUS > test_log_focus_txt_test_BART_MTL.log 2>&1
python evaluate_test_ppl.py --kp_method focus --model_name BART --model_checkpoint models/train_focus_BART_E2_L10_FOCUS > test_log_focus_txt_test_BART_MTL_ppl.log 2>&1

############################
# On full dataset - Cross Attention (our method)
############################
# train GPT2/BART
python train_focus.py --kp_method cross_attention --n_epochs 2 --lm_coef 10 --flag E2_L10_CROSSATTN --model_name GPT2 --incontext > train_log_crossattn_e2_L10_gpt.log 2>&1
python train_focus.py --kp_method cross_attention --n_epochs 2 --lm_coef 10 --flag E2_L10_CROSSATTN --model_name BART --incontext > train_log_crossattn_e2_L10_bart.log 2>&1


# test GPT2/BART
python evaluate_test.py --kp_method cross_attention --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E2_L10_CROSSATTN > test_log_crossattn_txt_test_GPT_MTL.log 2>&1
python evaluate_test_ppl.py --kp_method cross_attention --model_name GPT2 --model_checkpoint models/train_focus_GPT2_E2_L10_CROSSATTN > test_log_crossattn_txt_test_GPT_MTL_ppl.log 2>&1
python evaluate_test.py --kp_method cross_attention --model_name BART --model_checkpoint models/train_focus_BART_E2_L10_CROSSATTN > test_log_crossattn_txt_test_BART_MTL.log 2>&1
python evaluate_test_ppl.py --kp_method cross_attention --model_name BART --model_checkpoint models/train_focus_BART_E2_L10_CROSSATTN > test_log_crossattn_txt_test_BART_MTL_ppl.log 2>&1
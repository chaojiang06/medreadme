# Run the code
Please use the following command to 
```sh
python run_classification.py \
    --model_name_or_path roberta-large \
    --train_dataset_name_custom cwi.py \
    --train_split_name_custom train \
    --validation_dataset_name_custom cwi.py \
    --validation_split_name_custom validation \
    --test_dataset_name_custom cwi.py \
    --test_split_name_custom test \
    --shuffle_train_dataset \
    --label_column_name score \
    --metric_name pearsonr \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model pearsonr \
    --greater_is_better True \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir tmp \
    --save_total_limit 3 \
    --seed 1 \
    --report_to wandb \
    --run_name roberta-large+cwi.py+512+8+1e-5+1-20250901-1 \
    --logging_steps 10
```
Trained checkpoints are uploaded to the Hugging Face hub:

- [Best medical readability prediction model trained on our dataset.](https://huggingface.co/chaojiang06/medreadme_medical_sentence_readability_prediction_CWI)

# Reproduce
Please use the following command to reproduce [Table 7 in the paper](https://arxiv.org/pdf/2405.02144.pdf#page=8).
```sh
python run_classification.py \
    --model_name_or_path chaojiang06/medreadme_medical_sentence_readability_prediction_CWI \
    --train_dataset_name_custom cwi.py \
    --train_split_name_custom train \
    --validation_dataset_name_custom cwi.py \
    --validation_split_name_custom validation \
    --test_dataset_name_custom cwi.py \
    --test_split_name_custom test \
    --shuffle_train_dataset \
    --label_column_name score \
    --metric_name pearsonr \
    --do_eval \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --do_predict \
    --load_best_model_at_end \
    --metric_for_best_model pearsonr \
    --greater_is_better True \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --output_dir tmp \
    --save_total_limit 3 \
    --seed 1 \
    --report_to wandb \
    --run_name roberta-large+cwi.py+512+8+1e-5+1-20250901-1 \
    --logging_steps 10
```
The code will generate a `predict_results.txt` file in the output folder. You can then calculate the Pearson correlation for each source and average the results across different sources.

# Inference
If you want to use the code to do inference, the easiest way is to modify the test split of the `readability.csv`, add the sentences you want to predict, and populate with some random readability numbers, then `--do_predict` using the above command.

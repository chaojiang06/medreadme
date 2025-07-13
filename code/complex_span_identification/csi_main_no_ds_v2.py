


import sys
import os
import json
import argparse
import copy
import uuid
import shutil
from transformers import AutoTokenizer, AutoModelForTokenClassification
import datasets
from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, set_seed
import numpy as np
from modeling import RobertaForTokenClassificationAVG, BertaForTokenClassificationAVG
from csi_data_loader import Doc
import logging
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Features, Value, concatenate_datasets
from seqeval.metrics import classification_report, f1_score, recall_score, precision_score
from collections.abc import MutableMapping
import wandb
from dataclasses import dataclass, field
import dataclasses
import random

logger = logging.getLogger(__name__)

sys.path.insert(
    1, '/srv/nlprx-lab/share6/cjiang95/research_18_medical_cwi/src/WNUT_2020_NER/code/eval')

import evaluation

# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_WATCH"]="false"

metric = load_metric("seqeval")


def flatten_a_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_a_dict(
                value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def write_to_conll(tokens, true_preds, path):
    output = []
    
    with open(path, 'w', encoding='utf-8') as f:
        for t1, p1 in zip(tokens, true_preds):
            for t2, p2 in zip(t1, p1):
                f.write("{}\t{}\n".format(t2, p2))

            f.write("\n")
            
            
    # with open(path, 'w', encoding='utf-8') as f:
    #     f.writelines(output)

    logger.info(path)


def evaluate_with_jeniya_script(instances, gold_labels, pred_logits, granularity='fine'):
    # step 1, get tokens and prediction in two lists, make sure they have equal size
    tokens = [i['tokens'] for i in instances]
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in gold_labels]
    predictions = np.argmax(pred_logits, axis=-1)

    true_preds = []
    for l, p in zip(true_labels, predictions):
        true_preds.append([id2label[j] for j in p[:len(l)]])

    for t, p in zip(tokens, true_preds):
        assert len(t) == len(p), logger.error(
            "prediction and tokens have different length, {}, {}".format(t, p))
    # step 2, generate a random folder with two subfolders
    folder_name = "/tmp/{}".format(uuid.uuid1())
    while os.path.exists(folder_name):
        folder_name = "/tmp/{}".format(uuid.uuid1())
    pred_folder_name = '{}/pred/'.format(folder_name)
    gold_folder_name = '{}/gold/'.format(folder_name)
    os.mkdir(folder_name)
    os.mkdir(pred_folder_name)
    os.mkdir(gold_folder_name)
    # step 3, write the content into two folders

    if granularity == 'binary':
        for idx_i, i in enumerate(true_labels):
            for idx_j, j in enumerate(i):
                if j.startswith("B-"):
                    true_labels[idx_i][idx_j] = "B-jargon"
                elif j.startswith("I-"):
                    true_labels[idx_i][idx_j] = "I-jargon"

        for idx_i, i in enumerate(true_preds):
            for idx_j, j in enumerate(i):
                if j.startswith("B-"):
                    true_preds[idx_i][idx_j] = "B-jargon"
                elif j.startswith("I-"):
                    true_preds[idx_i][idx_j] = "I-jargon"

    elif granularity == 'middle':

        # cared_category = {
        #         'abbr-general',
        #         'abbr-medical',
        #         'general-complex',
        #         'general-medical-multisense',
        #         'medical-jargon-google-easy',
        #         'medical-jargon-google-hard',
        #         'medical-name-entity'
        #     }

        for idx_i, i in enumerate(true_labels):
            for idx_j, j in enumerate(i):
                if j.startswith("B-medical"):
                    true_labels[idx_i][idx_j] = "B-medical"
                elif j.startswith("I-medical"):
                    true_labels[idx_i][idx_j] = "I-medical"
                elif j.startswith("B-general"):
                    true_labels[idx_i][idx_j] = "B-general"
                elif j.startswith("I-general"):
                    true_labels[idx_i][idx_j] = "I-general"
                elif j.startswith("B-abbr"):
                    true_labels[idx_i][idx_j] = "B-abbr"
                elif j.startswith("I-abbr"):
                    true_labels[idx_i][idx_j] = "I-abbr"

        for idx_i, i in enumerate(true_preds):
            for idx_j, j in enumerate(i):
                if j.startswith("B-medical"):
                    true_preds[idx_i][idx_j] = "B-medical"
                elif j.startswith("I-medical"):
                    true_preds[idx_i][idx_j] = "I-medical"
                elif j.startswith("B-general"):
                    true_preds[idx_i][idx_j] = "B-general"
                elif j.startswith("I-general"):
                    true_preds[idx_i][idx_j] = "I-general"
                elif j.startswith("B-abbr"):
                    true_preds[idx_i][idx_j] = "B-abbr"
                elif j.startswith("I-abbr"):
                    true_preds[idx_i][idx_j] = "I-abbr"

    write_to_conll(tokens, true_preds, "{}/1.txt".format(pred_folder_name))
    write_to_conll(tokens, true_labels, "{}/1.txt".format(gold_folder_name))
    # step 4, call the evaluation script and return results
    results = evaluation.evaluate(
        input_gold_folder=gold_folder_name, input_pred_folder=pred_folder_name)

    # step 4, delete two folders
    shutil.rmtree(folder_name)
    return results


def convert_dict_to_datasetdict(split_to_examples):
    dataset = DatasetDict()
    # using your `Dict` object
    for k, v in split_to_examples.items():
        if len(v) == 0:
            continue
        tmp = {}
        keys = list(v[0].keys())
        for kk in keys:
            tmp[kk] = [i[kk] for i in v]
        dataset[k] = Dataset.from_dict(tmp)

    return dataset


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    logger.info(classification_report(
        true_labels, true_predictions, digits=3, output_dict=True))
    dict_fine_grained = classification_report(
        true_labels, true_predictions, digits=3, output_dict=True)

    dict_fine_grained_partial = classification_report(
        true_labels, true_predictions, digits=3, output_dict=True, partial_match=True)

    f1 = f1_score(
        true_labels, true_predictions)
    r = recall_score(
        true_labels, true_predictions)
    p = precision_score(
        true_labels, true_predictions)

    # I want to add binary performance for multi-label situation
    all_labels = set(flatten_list(true_labels))
    if "B-jargon" not in all_labels:
        logger.info("We are in the multi-label situation")

    for idx_i, i in enumerate(true_labels):
        for idx_j, j in enumerate(i):
            if j.startswith("B-medical"):
                true_labels[idx_i][idx_j] = "B-medical"
            elif j.startswith("I-medical"):
                true_labels[idx_i][idx_j] = "I-medical"
            elif j.startswith("B-general"):
                true_labels[idx_i][idx_j] = "B-general"
            elif j.startswith("I-general"):
                true_labels[idx_i][idx_j] = "I-general"
            elif j.startswith("B-abbr"):
                true_labels[idx_i][idx_j] = "B-abbr"
            elif j.startswith("I-abbr"):
                true_labels[idx_i][idx_j] = "I-abbr"

    for idx_i, i in enumerate(true_predictions):
        for idx_j, j in enumerate(i):
            if j.startswith("B-medical"):
                true_predictions[idx_i][idx_j] = "B-medical"
            elif j.startswith("I-medical"):
                true_predictions[idx_i][idx_j] = "I-medical"
            elif j.startswith("B-general"):
                true_predictions[idx_i][idx_j] = "B-general"
            elif j.startswith("I-general"):
                true_predictions[idx_i][idx_j] = "I-general"
            elif j.startswith("B-abbr"):
                true_predictions[idx_i][idx_j] = "B-abbr"
            elif j.startswith("I-abbr"):
                true_predictions[idx_i][idx_j] = "I-abbr"

    logger.info('this is the performance using middle-level evaluation')
    logger.info(classification_report(
        true_labels, true_predictions, digits=3, output_dict=True))
    dict_middle = classification_report(
        true_labels, true_predictions, digits=3, output_dict=True)
    dict_middle_partial = classification_report(
        true_labels, true_predictions, digits=3, output_dict=True, partial_match=True)

    for idx_i, i in enumerate(true_labels):
        for idx_j, j in enumerate(i):
            if j.startswith("B-"):
                true_labels[idx_i][idx_j] = "B-jargon"
            elif j.startswith("I-"):
                true_labels[idx_i][idx_j] = "I-jargon"

    for idx_i, i in enumerate(true_predictions):
        for idx_j, j in enumerate(i):
            if j.startswith("B-"):
                true_predictions[idx_i][idx_j] = "B-jargon"
            elif j.startswith("I-"):
                true_predictions[idx_i][idx_j] = "I-jargon"
    logger.info('this is the performance using binary evaluation')
    logger.info(classification_report(
        true_labels, true_predictions, digits=3, output_dict=True))
    dict_coarse = classification_report(
        true_labels, true_predictions, digits=3, output_dict=True)
    dict_coarse_partial = classification_report(
        true_labels, true_predictions, digits=3, output_dict=True, partial_match=True)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        **flatten_a_dict(dict_fine_grained, parent_key='dict_fine_grained'),
        **flatten_a_dict(dict_fine_grained_partial, parent_key='dict_fine_grained_partial'),
        **flatten_a_dict(dict_coarse, parent_key='dict_coarse'),
        **flatten_a_dict(dict_coarse_partial, parent_key='dict_coarse_partial'),
        **flatten_a_dict(dict_middle, parent_key='dict_middle'),
        **flatten_a_dict(dict_middle_partial, parent_key='dict_middle_partial'),
    }


def remap_word_ids(word_ids):
    # special tokens or paddings are map to 0
    word_idx = []
    for i in word_ids:
        if i == None:
            word_idx.append(0)
        else:
            word_idx.append(i + 1)
    return word_idx


def process_ner_labels_binary(examples):

    tokens = examples["tokens"]

    entity_mentions = examples["entities"]
    labels = [0 for _ in range(len(tokens))]
    for ent in entity_mentions:
        start, end = ent[0], ent[1]
        for idx in range(start, end):
            if idx == start:
                labels[idx] = entity_label["B-jargon"]
            else:
                labels[idx] = entity_label["I-jargon"]
    examples["ner_tags"] = labels
    return examples


def process_ner_labels_multi(examples):

    # cared_category = {
    #         'abbr-general',
    #         'abbr-medical',
    #         'general-complex',
    #         'general-medical-multisense',
    #         'medical-jargon-google-easy',
    #         'medical-jargon-google-hard',
    #         'medical-name-entity'
    #     }

    #     'train': [
    #         {'tokens': i['tokens'],
    #          'sent_id': idx_i,
    #          'entities': [[j[0], j[1]] for j in i['entities']],
    #          'entities_type': [j[2] for j in i['entities']],
    #          'entities_tokens': [j[3] for j in i['entities']],
    #          'sent_raw_id': "{}|||{}|||".format(i['belongings'][0], i['belongings'][1], i['file_path'])}
    #         for idx_i, i in enumerate(train_annotations)
    #     ],

    tokens = examples["tokens"]
    entity_mentions = examples["entities"]
    tags = examples["entities_type"]

    # + examples["agents"] + \
    #     examples["patients"] + examples["money"]
    # tags = ['ANCHOR'] * len(examples["anchors"]) + ['AGENT'] * \
    #     len(examples["agents"]) + ['PATIENT'] * \
    #     len(examples["patients"]) + ['MONEY'] * len(examples["money"])
    labels = [0 for _ in range(len(tokens))]
    for idx_ent, ent in enumerate(entity_mentions):
        start, end = ent[0], ent[1]
        tag = tags[idx_ent]
        for idx in range(start, end):
            if idx == start:
                labels[idx] = entity_label["B-" + tag]
            else:
                labels[idx] = entity_label["I-" + tag]

    examples['tokens'] = tokens
    examples["ner_tags"] = labels

    assert len(examples["tokens"]) == len(examples["ner_tags"])

    return examples


def tokenize_and_align_labels(examples):

    all_tokens = examples["tokens"]
    all_labels = examples["ner_tags"]

    for idx in range(len(all_tokens)):
        assert len(all_tokens[idx]) == len(all_labels[idx])
        tmp_tokens = []
        tmp_labels = []

        for idx_token, token in enumerate(all_tokens[idx]):
            # if token != '\u200c':
            tmp_tokens.append(token)
            tmp_labels.append(all_labels[idx][idx_token])

        all_tokens[idx] = tmp_tokens
        all_labels[idx] = tmp_labels

    tokenized_inputs = tokenizer(
        all_tokens, truncation=True, is_split_into_words=True, max_length=256,
    )

    # all_labels = examples["ner_tags"]
    # input_ids = tokenized_inputs["input_ids"]
    word_idx = []
    # new_label_list = []
    for i, labels in enumerate(all_labels):

        new_word_ids = remap_word_ids(tokenized_inputs.word_ids(i))
        word_idx.append(new_word_ids)
        # new_label = labels + [-100] * (len(input_ids[i]) - len(labels))
        # assert len(new_label) == len(input_ids[i])
        # new_label_list.append(new_label)
    # print(word_idx)
    # print(tokenized_inputs)
    tokenized_inputs["token_type_ids"] = word_idx
    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs


def predict_anchor(model_name_or_path, load_dir, raw_dataset, split):
    # this function should be same as the one outside.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True, max_length=256,
        )

        all_labels = examples["ner_tags"]
        input_ids = tokenized_inputs["input_ids"]
        word_idx = []
        new_label_list = []
        for i, labels in enumerate(all_labels):

            new_word_ids = remap_word_ids(tokenized_inputs.word_ids(i))
            word_idx.append(new_word_ids)
            # new_label = labels + [-100] * (len(input_ids[i]) - len(labels))
            # assert len(new_label) == len(input_ids[i])
            # new_label_list.append(new_label)
        tokenized_inputs["token_type_ids"] = word_idx
        tokenized_inputs["labels"] = all_labels

        return tokenized_inputs

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # datasets_ner = raw_dataset.map(process_ner_labels)
    
    if model_name_or_path.lower() in ['roberta-base', 'roberta-large']:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if input_args.train_coarse == "True":
        datasets_ner = dataset.map(process_ner_labels_binary)
    else:
        datasets_ner = dataset.map(process_ner_labels_multi)



    tokenized_datasets = datasets_ner.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=datasets_ner[split].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    if 'roberta' in model_name_or_path.lower():
        model = RobertaForTokenClassificationAVG.from_pretrained(
            load_dir,
            id2label=id2label,
            label2id=label2id,
            output_hidden_states=True,
            classifier_dropout=input_args.classifier_dropout,
        )
    elif 'bert' in model_name_or_path.lower():
        model = BertaForTokenClassificationAVG.from_pretrained(
            load_dir,
            id2label=id2label,
            label2id=label2id,
            output_hidden_states=True,
            classifier_dropout=input_args.classifier_dropout,
        )
    else:
        logger.error("model name not supported")

    args = TrainingArguments(
        load_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,  # 2e-5
        num_train_epochs=30,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=1,
        save_total_limit=5,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        # deepspeed='ds_zero0.json'
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    predictions, labels, metrics = trainer.predict(
        tokenized_datasets[split], metric_key_prefix=split)

    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_predictions


def from_path_to_split(train_folder_paths):
    # step 2, load the raw data for each split
    train_data = []
    for train_folder_path in train_folder_paths:
        for i in range(15):
            train_file_path = train_folder_path + str(i)
            train_data.append(Doc(train_file_path))

    # step 2.5, do some further processing, make sure the sentences and annotations make senses.
    # step 2.5.1, load all sentences
    train_lines = []
    for i in train_data:
        for line in i.lines:
            if line.belongings[0] != 'meta-data':
                train_lines.append(line)
    # step 2.5.2, load the annotations for each sentences, probably only use the unmodified part?
    train_annotations = []
    for line in train_lines:
        res = line.generate_instance()
        if res['modified'] == False:
            train_annotations.append(line.generate_instance())
        else:
            train_annotations.append(line.generate_instance())
            logger.warning("modified sentence found")

    # step 2.5.3, filter very short sentences and remove '' at the end of a sentence, at the same time, judge no entities are annotated on top of it
    train_annotations_tmp = []
    for i in train_annotations:

        # remove the ones with <= 4 REAL tokens
        tmp = " ".join(i['tokens']).strip().rstrip().split()
        if len(tmp) <= 3:
            continue

        # remove the last '' in a sentence, and make sure no entities are annotated on top of it
        if i['tokens'][-1] == '':
            for j in i['entities']:
                assert len(i['tokens']) - 1 >= j[1], logger.error(
                    "entity is annotated on top of the white space: {}".format(i))

            i['tokens'] = i['tokens'][:-1]

        train_annotations_tmp.append(i)

    return train_annotations_tmp


train_folder_paths = "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-5-chao-v3/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-7-merge-pass-1/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-8-mithun/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-9-mithun/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-10-mithun/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-11-mithun/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-12-mithun/"
dev_folder_paths = "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-13-mithun/;" \
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-14-mithun/"
    
test_folder_paths = "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-15-mithun/;"\
    "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-16-mithun/;" \
     "/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-17-mithun/"
                    

# take args
parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--train_data", default='medcsi', type=str)
parser.add_argument("--train_split", default=train_folder_paths, type=str)
parser.add_argument("--dev_split", default=dev_folder_paths, type=str)
parser.add_argument("--test_split", default=test_folder_paths, type=str)
parser.add_argument("--load_ckpt_at_begining", default='None', type=str)
parser.add_argument("--model_name_or_path",
                    default='bert-large-cased', type=str)
parser.add_argument("--classifier_dropout", default=0.2, type=float)
parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--per_device_train_batch_size", default=32, type=int)
parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
parser.add_argument("--num_train_epochs", default=20, type=int)
parser.add_argument("--warmup_ratio", default=0.00, type=float)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--output_dir", default='outputs/anchor-c', type=str,
                    help="The output directory where the model predictions will be written.")
parser.add_argument("--seed", default=0000, type=int)
parser.add_argument("--train_coarse", default="False")
parser.add_argument("--train_portion", default=100, type = int)
parser.add_argument("--run_name", default='test', type = str)


if __name__ == '__main__':
    input_args = parser.parse_args()
    set_seed(input_args.seed)
    logger.setLevel(logging.INFO)

    # make a dir for input_args.output_dir
    os.makedirs(input_args.output_dir, exist_ok=True)

    # Create a file handler to write log messages to a file
    file_handler = logging.FileHandler(
        os.path.join(input_args.output_dir, 'logger.json'))
    # Set the logging level for this handler
    file_handler.setLevel(logging.INFO)

    # Create a console handler to write log messages to the console
    console_handler = logging.StreamHandler()
    # Set the logging level for this handler
    console_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the file and console handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if input_args.train_coarse == "True":
        entity_label = {'O': 0, 'B-jargon': 1, 'I-jargon': 2}
    else:
        cared_category = {
            'abbr-general',
            'abbr-medical',
            'general-complex',
            'general-medical-multisense',
            'medical-jargon-google-easy',
            'medical-jargon-google-hard',
            'medical-name-entity'
        }
        entity_label = {'O': 0,
                        'B-abbr-general': 1,
                        'I-abbr-general': 2,
                        'B-abbr-medical': 3,
                        'I-abbr-medical': 4,
                        'B-general-complex': 5,
                        'I-general-complex': 6,
                        'B-general-medical-multisense': 7, 'I-general-medical-multisense': 8, 'B-medical-jargon-google-easy': 9, 'I-medical-jargon-google-easy': 10,
                        'B-medical-jargon-google-hard': 11, 'I-medical-jargon-google-hard': 12,
                        'B-medical-name-entity': 13,
                        'I-medical-name-entity': 14}

    label2id = entity_label
    id2label = {v: k for k, v in entity_label.items()}

    # here I will need to load the data in specific format
    # step 1, set train/dev/test split

    train_folder_paths = input_args.train_split.split(';')
    
    # train_folder_paths =  train_folder_paths[:input_args.train_portion]
    
    dev_folder_paths = input_args.dev_split.split(';')
    test_folder_paths = input_args.test_split.split(';')

    train_annotations = from_path_to_split(train_folder_paths)
    
    logger.info("number of original total training examples: {}".format(len(train_annotations)))
    
    number_of_elements = int(len(train_annotations) * input_args.train_portion / 100)
    train_annotations =  random.sample(train_annotations, number_of_elements)
    
    logger.info("train portion: {}%".format(input_args.train_portion))
    logger.info("number of training examples: {}".format(len(train_annotations)))
    
        
    dev_annotations = from_path_to_split(dev_folder_paths)
    test_annotations = from_path_to_split(test_folder_paths)

    # step 3, convert to a dict format

    dataset = {
        'train': [
            {'tokens': i['tokens'],
             'sent_id': idx_i,
             'entities': [[j[0], j[1]] for j in i['entities']],
             'entities_type': [j[2] for j in i['entities']],
             'entities_tokens': [j[3] for j in i['entities']],
             'sent_raw_id': "{}|||{}|||".format(i['belongings'][0], i['belongings'][1], i['file_path'])}
            for idx_i, i in enumerate(train_annotations)
        ],
        'dev': [
            {'tokens': i['tokens'],
             'sent_id': idx_i,
             'entities': [[j[0], j[1]] for j in i['entities']],
             'entities_type': [j[2] for j in i['entities']],
             'entities_tokens': [j[3] for j in i['entities']],
             'sent_raw_id': "{}|||{}|||".format(i['belongings'][0], i['belongings'][1], i['file_path'])}
            for idx_i, i in enumerate(dev_annotations)
        ],
        'test': [
            {'tokens': i['tokens'],
             'sent_id': idx_i,
             'entities': [[j[0], j[1]] for j in i['entities']],
             'entities_type': [j[2] for j in i['entities']],
             'entities_tokens': [j[3] for j in i['entities']],
             'sent_raw_id': "{}|||{}|||".format(i['belongings'][0], i['belongings'][1], i['file_path'])}
            for idx_i, i in enumerate(test_annotations)
        ],
    }

    # step 4, convert to a dataset format
    dataset = convert_dict_to_datasetdict(dataset)

    logger.info(dataset)
    wandb.init(
        project="cwi-span",
        name=input_args.run_name,
    )

    wandb.config.update({"custom_args": vars(input_args)})
    
    # step 5, go through process_ner_labels
    model_name_or_path = input_args.model_name_or_path

    if model_name_or_path.lower() in ['roberta-base', 'roberta-large']:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if input_args.train_coarse == "True":
        datasets_ner = dataset.map(process_ner_labels_binary)
    else:
        datasets_ner = dataset.map(process_ner_labels_multi)

    # step 6, go through tokenize_and_align_labels
    tokenized_datasets = datasets_ner.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=datasets_ner["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    if 'roberta' in model_name_or_path.lower():
        model = RobertaForTokenClassificationAVG.from_pretrained(
            model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            output_hidden_states=True,
            classifier_dropout=input_args.classifier_dropout,
        )
    elif 'bert' in model_name_or_path.lower():
        model = BertaForTokenClassificationAVG.from_pretrained(
            model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            output_hidden_states=True,
            classifier_dropout=input_args.classifier_dropout,
        )
    else:
        logger.error("model name not supported")

    logger.info(model)

    save_path = input_args.output_dir

    args = TrainingArguments(
        save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=input_args.learning_rate,  # 2e-5
        num_train_epochs=input_args.num_train_epochs,
        weight_decay=input_args.weight_decay,
        per_device_train_batch_size=input_args.per_device_train_batch_size,
        per_device_eval_batch_size=input_args.per_device_eval_batch_size,
        warmup_ratio=input_args.warmup_ratio,
        save_total_limit=5,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        # deepspeed='ds_zero0.json'
        report_to = 'wandb'
    )
    
    wandb.config.update({"training_args": dataclasses.asdict(args)})

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    if input_args.load_ckpt_at_begining != 'None':
        trainer.train(resume_from_checkpoint=input_args.load_ckpt_at_begining)
    else:
        trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    metric_dev = trainer.evaluate(
        tokenized_datasets['dev'], metric_key_prefix="dev")
    metric_test = trainer.evaluate(
        tokenized_datasets['test'], metric_key_prefix="test")

    dev_pred = trainer.predict(
        tokenized_datasets['dev'], metric_key_prefix="dev")
    test_pred = trainer.predict(
        tokenized_datasets['test'], metric_key_prefix="test")

    dev_jeniya_script = evaluate_with_jeniya_script(
        dev_annotations, dev_pred.label_ids, dev_pred.predictions)
    test_jeniya_script = evaluate_with_jeniya_script(
        test_annotations, test_pred.label_ids, test_pred.predictions)

    trainer.log_metrics("dev", metric_dev)
    trainer.log_metrics("test", metric_test)
    trainer.log_metrics("dev_jeniya_script", flatten_a_dict(dev_jeniya_script))
    trainer.log_metrics("test_jeniya_script",
                        flatten_a_dict(test_jeniya_script))

    for k, v in metric_dev.items():
            wandb.run.summary["{}/{}".format('metric_dev', k)] = v
    for k, v in metric_test.items():
            wandb.run.summary["{}/{}".format('metric_test', k)] = v
    for k, v in flatten_a_dict(dev_jeniya_script).items():
            wandb.run.summary["{}/{}".format('dev_jeniya_script', k)] = v
    for k, v in flatten_a_dict(test_jeniya_script).items():
            wandb.run.summary["{}/{}".format('test_jeniya_script', k)] = v

    output_log = {
        "dev": metric_dev,
        "test": metric_test,
        "dev_jeniya_script": dev_jeniya_script,
        "test_jeniya_script": test_jeniya_script,
        'input_args': vars(input_args),
    }

    if input_args.train_coarse != "True":
        dev_jeniya_script_coarse = evaluate_with_jeniya_script(
            dev_annotations, dev_pred.label_ids, dev_pred.predictions, granularity = 'binary')
        test_jeniya_script_coarse = evaluate_with_jeniya_script(
            test_annotations, test_pred.label_ids, test_pred.predictions, granularity = 'binary')

        trainer.log_metrics("dev_jeniya_script_coarse",
                            flatten_a_dict(dev_jeniya_script_coarse))
        for k, v in flatten_a_dict(dev_jeniya_script_coarse).items():
            wandb.run.summary["{}/{}".format('dev_jeniya_script_coarse', k)] = v
        
        trainer.log_metrics("test_jeniya_script_coarse",
                            flatten_a_dict(test_jeniya_script_coarse))
        for k, v in flatten_a_dict(test_jeniya_script_coarse).items():
            wandb.run.summary["{}/{}".format('test_jeniya_script_coarse', k)] = v

        output_log['dev_jeniya_script_coarse'] = dev_jeniya_script_coarse
        output_log['test_jeniya_script_coarse'] = test_jeniya_script_coarse
        
        dev_jeniya_script_middle = evaluate_with_jeniya_script(
            dev_annotations, dev_pred.label_ids, dev_pred.predictions, granularity = 'middle')
        test_jeniya_script_middle = evaluate_with_jeniya_script(
            test_annotations, test_pred.label_ids, test_pred.predictions, granularity = 'middle')

        trainer.log_metrics("dev_jeniya_script_middle",
                            flatten_a_dict(dev_jeniya_script_middle))
        for k, v in flatten_a_dict(dev_jeniya_script_middle).items():
            wandb.run.summary["{}/{}".format('dev_jeniya_script_middle', k)] = v
        trainer.log_metrics("test_jeniya_script_middle",
                            flatten_a_dict(test_jeniya_script_middle))
        for k, v in flatten_a_dict(test_jeniya_script_middle).items():
            wandb.run.summary["{}/{}".format('test_jeniya_script_middle', k)] = v

        output_log['dev_jeniya_script_middle'] = dev_jeniya_script_middle
        output_log['test_jeniya_script_middle'] = test_jeniya_script_middle
        

    with open(os.path.join(save_path, 'output_log.json'), 'w') as f:
        json.dump(output_log, f, indent=2)

    
    output_prediction = {
        'train':{
            'tokens': dataset['train']['tokens'],
            'sent_id': dataset['train']['sent_id'],
            'entities': dataset['train']['entities'],
            'entities_type': dataset['train']['entities_type'],
            'entities_tokens': dataset['train']['entities_tokens'],
            'sent_raw_id': dataset['train']['sent_raw_id'],
            'prediction': predict_anchor(model_name_or_path, input_args.output_dir, dataset, 'train'),
        },
        'dev':{
            'tokens': dataset['dev']['tokens'],
            'sent_id': dataset['dev']['sent_id'],
            'entities': dataset['dev']['entities'],
            'entities_type': dataset['dev']['entities_type'],
            'entities_tokens': dataset['dev']['entities_tokens'],
            'sent_raw_id': dataset['dev']['sent_raw_id'],
            'prediction': predict_anchor(model_name_or_path, input_args.output_dir, dataset, 'dev'),
        },
        'test':{
            'tokens': dataset['test']['tokens'],
            'sent_id': dataset['test']['sent_id'],
            'entities': dataset['test']['entities'],
            'entities_type': dataset['test']['entities_type'],
            'entities_tokens': dataset['test']['entities_tokens'],
            'sent_raw_id': dataset['test']['sent_raw_id'],
            'prediction': predict_anchor(model_name_or_path, input_args.output_dir, dataset, 'test'),
        },
    }
    
    with open(os.path.join(save_path, 'output_prediction.json'), 'w') as f:
        json.dump(output_prediction, f, indent=2)
    
    print("done")

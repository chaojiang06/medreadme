# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3


import csv
import os
import pandas as pd
import datasets
import sys


# _DOWNLOAD_URL = "https://raw.githubusercontent.com/timpal0l/sts-benchmark-swedish/master/data/stsb-mt-sv.zip"
# _TRAIN_FILE = "train-sv.tsv"
# _VAL_FILE = "dev-sv.tsv"
# _TEST_FILE = "test-sv.tsv"

# _BASE_FOLDER = "/nethome/cjiang95/share6/research_18_medical_cwi/data/readme++/"

# _ALL_FILE = "readme-english.xlsx"

# _CITATION = """\
# @article{naous2023towards,
#   title={Towards Massively Multi-domain Multilingual Readability Assessment},
#   author={Naous, Tarek and Ryan, Michael J and Chandra, Mohit and Xu, Wei},
#   journal={arXiv preprint arXiv:2305.14463},
#   year={2023}
# }
# """

# _DESCRIPTION = "Towards Massively Multi-domain Multilingual Readability Assessment"




# step 1, load all sentences into a dataframe, has column of source and complex simple side, has column of article index
# step 2, load all readability scores into a dict
# step 3, fill the readability score into the dataframe

def load_raw_data():

    data_df = pd.read_csv("../../dataset/readability.csv")
    return data_df



class StsbMtSv(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "sentence": datasets.Value("string"),
                    # "domain": datasets.Value("string"),
                    # "subdomain": datasets.Value("string"),
                    # "paragraph": datasets.Value("string"),
                    # "context": datasets.Value("string"),
                    "score": datasets.Value("float"),
                }
            ),
            supervised_keys=None,
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_dir = dl_manager.download_and_extract(_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test"},
            ),
            
        ]

    def _generate_examples(self, split):
        # """This function returns the examples in the raw (text) form."""
        # with open(filepath, encoding="utf-8") as f:
        #     reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        #     for idx, row in enumerate(reader):
        #         yield idx, {
        #             "sentence": row["sentence"],
        #             "score1": row["score1"],
        #             "score2": row["score2"],
        #             "score": row["score"],
        #         }
        data_df =  load_raw_data()
        d = data_df[data_df['split'] == split]
        for idx, row in d.iterrows():
            yield idx, {
                "sentence": row['Sentence'].encode('utf-8'),
                # "domain": row['Domain'],
                # "subdomain": row['Sub-domain'],
                # "paragraph": row['Paragraph'],
                # "context": row['Context'],
                "score": row['Readability']
            }

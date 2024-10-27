# MedReadMe: A Systematic Study for Fine-grained Sentence Readability in Medical Domain

Our paper is accepted by EMNLP 2024 main conference as an oral presentation. The paper is available at [arXiv](https://arxiv.org/abs/2405.02144).

## Dataset
The readability and jargon datasets are in the `dataset` folder.

Readability folder can be load using the following code snippet:
```python
import pandas as pd
readability = pd.read_csv('dataset/readability.csv')
```

The jargon annotation can be load using the following code snippet:
```python
import json
jargon = json.load(open('dataset/jargon.json'))
```

## Citation
If you use this dataset, please cite the following paper:
```
@article{Jiang2024MedReadMeAS,
  title={MedReadMe: A Systematic Study for Fine-grained Sentence Readability in Medical Domain},
  author={Chao Jiang and Wei Xu},
  journal={ArXiv},
  year={2024},
  volume={abs/2405.02144},
  url={https://api.semanticscholar.org/CorpusID:269587954}
}
```


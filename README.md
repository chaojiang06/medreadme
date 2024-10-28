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
@article{jiang2024medreadmesystematicstudyfinegrained,
      title={MedReadMe: A Systematic Study for Fine-grained Sentence Readability in Medical Domain}, 
      author={Chao Jiang and Wei Xu},
      year={2024},
      eprint={2405.02144},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.02144}, 
}
```
## Acknowledgement
This research is supported in part by the NSF CAREER Award IIS-2144493, NSF Award IIS-2112633, NIH Award R01LM014600, ODNI and IARPA via the HIATUS program (contract 2022-22072200004). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of NSF, NIH, ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.


import logging
import os
import csv
import copy
import shutil
import numpy as np
import bisect

logger = logging.getLogger(__name__)


# a function to read the json file
import json, pickle
import random
from tqdm.notebook import tqdm

def read_json_file(path):
    with open(path) as f:
        data = json.load(f)
        
    return data

def read_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    return data

# def read_txt_file(path):
#     with open(path) as f:
#         data = f.readlines()
#     data = [x.strip() for x in data]
#     return data

class Entity:
  def __init__(self, index, anno_type, start, end, text):
    self.index = index
    
    self.label = anno_type

    self.string = text
    self.tokens = text.split()
    
    self.defined = False
    self.defined_index = None
    
    self.elaborated = False
    self.elaborated_index = None

    self.char_index_start = int(start)
    self.char_index_end = int(end)
    
    self.token_index_start = None
    self.token_index_end = None
    
    assert self.char_index_end == self.char_index_start + len(self.string), f'char_index_end: {self.char_index_end}, char_index_start: {self.char_index_start}, len(self.string): {len(self.string)}, self.string: A{self.string}A'
    
  def __eq__(self, other):
    if self.label == other.label and self.char_index_start == other.char_index_start and self.char_index_end == other.char_index_end and self.string == other.string and \
       self.defined == other.defined and self.elaborated == other.elaborated:
      return True
    else:
      return False
    
  def __str__(self):
    return f'Entity({self.index}, {self.label}, {self.char_index_start}, {self.char_index_end}, {self.string}), token_index_start: {self.token_index_start}, token_index_end: {self.token_index_end}'
    

  def __repr__(self):
    return self.__str__()

Entity(0, 'google-hard', 0, 4, 'test')

class Token:
  def __init__(self, paragraph_index, sentence_index, token_index, accumulate_start_offset, accumulate_end_offset, token_string):
    self.paragraph_index = int(paragraph_index)
    self.sentence_index = int(sentence_index)
    self.token_index = int(token_index)
    self.accumulate_start_offset = int(accumulate_start_offset)
    self.accumulate_end_offset = int(accumulate_end_offset)
    self.string = token_string
    
  def __str__(self):
    return f'Token({self.paragraph_index}, {self.sentence_index}, {self.token_index}, {self.accumulate_start_offset}, {self.accumulate_end_offset}, {self.string})'
  
  def __repr__(self):
    return self.__str__()
  
Token(0, 0, 0, 0, 4, 'test')

cared_category = {
    'abbr-general',
    'abbr-medical',
    'general-complex',
    'general-medical-multisense',
    'medical-jargon-google-easy',
    'medical-jargon-google-hard',
    'medical-name-entity'
 }

class Line:
  def __init__(self, paragraph_index, sentence_index, accumulate_start_offset, accumulate_end_offset, sentence_string,file_path):
    self.paragraph_index = int(paragraph_index)
    self.sentence_index = int(sentence_index)
    self.accumulate_start_offset = int(accumulate_start_offset)
    self.accumulate_end_offset = int(accumulate_end_offset)
    self.string = sentence_string
    self.tokens = []
    self.entities = []
    self.belongings = []
    self.file_path = file_path
    
    assert self.accumulate_end_offset == self.accumulate_start_offset + len(self.string), f'accumulate_end_offset: {self.accumulate_end_offset}, accumulate_start_offset: {self.accumulate_start_offset}, len(self.string): {len(self.string)}, self.string: {self.string}'
  
  def check_entity(self, entity):
    if entity.char_index_start >= self.accumulate_start_offset and entity.char_index_end <= self.accumulate_end_offset:
      return True
    else:
      return False
    
  def add_entity(self, entity):
    self.entities.append(entity)
    
  def __str__(self):
    return f'Line({self.paragraph_index}, {self.sentence_index}, {self.accumulate_start_offset}, {self.accumulate_end_offset}, {self.string}, {self.tokens}, {self.belongings})'
  
  def __repr__(self):
    return self.__str__()
  
  def check_cover_full_token(self):
    # for every entity, its start and end should match something with the tokens
    all_accumulate_start_offset_for_tokens = [entity.accumulate_start_offset for entity in self.tokens]
    all_accumulate_end_offset_for_tokens = [entity.accumulate_end_offset for entity in self.tokens]
    
    for entity in self.entities:
      if entity.char_index_start not in all_accumulate_start_offset_for_tokens or \
         entity.char_index_end not in all_accumulate_end_offset_for_tokens:
        
        logging.warning(f'find something mismatched: {entity}')
        
        # print(all_accumulate_start_offset_for_tokens)
        # print(bisect.bisect_right(all_accumulate_start_offset_for_tokens, entity.char_index_start))
        
        entity.char_index_start = all_accumulate_start_offset_for_tokens[bisect.bisect_right(all_accumulate_start_offset_for_tokens, entity.char_index_start) -1 ]
        entity.char_index_end = all_accumulate_end_offset_for_tokens[bisect.bisect_left(all_accumulate_end_offset_for_tokens, entity.char_index_end)]
        entity.string = self.string[entity.char_index_start - self.accumulate_start_offset:entity.char_index_end - self.accumulate_start_offset]
        entity.tokens = entity.string.split()
        logging.debug(f'the sentence: {self.string}')
        logging.debug(f'the new string: {entity.string}')
        logging.warning(f'after fix: {entity}')
        
    for entity in self.entities:
      tokens = [(token.accumulate_start_offset, token.accumulate_end_offset) for token in self.tokens]
      covered_token_indices = [i for i, token in enumerate(tokens) if entity.char_index_start <= token[0] and entity.char_index_end >= token[1]]
      
      entity.token_index_start = covered_token_indices[0]
      entity.token_index_end = covered_token_indices[-1] + 1
      

    
    
  def generate_instance(self):
    # step 1, find and remove any nesting entities
  
    
    FLAG_whole = False
    
    
    entities = [i for i in self.entities if i.label in cared_category]
    entities = sorted(entities, key=lambda x: (x.char_index_end, x.char_index_start))
    
    entities_tmp0 = []
    for idx_i, i in enumerate(entities):
      for idx_j, j in enumerate(entities):
        FLAG = True
        if idx_i == idx_j:
          continue
        if i.char_index_start >= j.char_index_start and i.char_index_end <= j.char_index_end:
          FLAG = False
          FLAG_whole = True
          break
      entities_tmp0.append(i) if FLAG else logging.warning(f'find nesting entity: {i} and {j} from line {self}')
    
    
    
    entities_tmp0 = sorted(entities_tmp0, key=lambda x: (x.char_index_end, x.char_index_start))
    entities_tmp1 = []
    for idx_i, i in enumerate(entities_tmp0):
      FLAG = True
      for idx_j, j in enumerate(entities_tmp0):
        if idx_i == idx_j:
          continue
        
        if idx_i < idx_j:
          continue
        
        if i.char_index_start >= j.char_index_start and i.char_index_end <= j.char_index_end:
          FLAG = False
          FLAG_whole = True
          break
        
      entities_tmp1.append(i) if FLAG else logging.warning(f'find nesting entity: {i} and {j} from line {self}')
      
    entities_tmp1  = sorted(entities_tmp1, key=lambda x: (x.char_index_end, x.char_index_start))
    # step 3, remove overlapping entities
    entities_tmp2 = []
    for idx_i, i in enumerate(entities_tmp1):
      FLAG = True
      for idx_j, j in enumerate(entities_tmp1):
        if idx_i == idx_j:
          continue
        if idx_i < idx_j:
          continue
        if i.char_index_start < j.char_index_end:
          FLAG = False
          FLAG_whole = True
          break
        
      entities_tmp2.append(i) if FLAG else logging.warning(f'find overlapping entity: {i} and {j} from line {self}')

      entities_tmp2  = sorted(entities_tmp2, key=lambda x: (x.char_index_end, x.char_index_start))
      # step 3, final check
      for idx_i, i in enumerate(entities_tmp2):
        for idx_j, j in enumerate(entities_tmp2):
          if idx_i == idx_j: continue
          if idx_i < idx_j: continue
          assert i.char_index_start >= j.char_index_end, f'find overlapping entity: {i} and {j} from line {self}'
    
    result = {
      "tokens":[ k.string for k in self.tokens],
      'entities': [[k.token_index_start, k.token_index_end, k.label, k.tokens] for k in entities_tmp2],
      'modified': FLAG_whole,
      'belongings': self.belongings,
      'file_path': self.file_path,
    }
    
    logger.debug(f'after generate instance: {result}')
    
    return result
    
    
    
    
    
  
a = Line(0, 0, 0, 18, 'test1 test2 test3\n', "path")
a.add_entity(Entity(0, 'medical-jargon-google-hard', 0, 17, 'test1 test2 test3'))
a.add_entity(Entity(0, 'medical-jargon-google-hard', 6, 11, 'test2'))
a.tokens = [Token(0, 0, 0, 0, 5, 'test1'), Token(0, 0, 1, 6, 11, 'test2'), Token(0, 0, 2, 12, 17, 'test3')]
print(a.entities)
print(a.tokens)

a.check_cover_full_token()
b = a.generate_instance() 
# print(a.generate_instance())

print(b)

def read_tsv_file(path):
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        return list(reader)

def read_txt_file(path):
    with open(path, encoding='utf-8') as f:
        data = f.readlines()
        
    return data

def read_ann_file(path):
    data = read_txt_file(path)
    data = [i.strip().split("\t") for i in data]
    
    # print(path)
    
    entities = [i for i in data if i[0].startswith("T")]
    attributes = [i for i in data if i[0].startswith("A")]
    # print(entities)
    assert all(len(x) == 3 for x in entities), print([i for i in entities if len(i) != 3])
    entities = [[i[0]] + i[1].split(" ") + [i[2]] for i in entities]
    assert all(len(x) == 5 for x in entities), print([i for i in entities if len(i) != 5])   
    # print(entities) 
    entities = [Entity(*i) for i in entities]
    
    assert all(len(x) == 2 for x in attributes), print([i for i in attributes if len(i) != 2])
    attributes = [[i[0]] + i[1].split(" ") for i in attributes]
    assert all(len(x) == 3 for x in attributes), print([i for i in attributes if len(i) != 3])
    
    for i in attributes:
        for j in entities:
            if i[2] == j.index:
                if i[1] == "defined":
                    j.defined = True
                    j.defined_index = i[0]
                elif i[1] == "elaborated":
                    j.elaborated = True
                    j.elaborated_index = i[0]
                else:
                    raise Exception("Unknown attribute: " + i[1])
    
    return entities


def parse_txt_file(path):
    data = read_txt_file(path)
    """
    for every line, I want to know its paragraph index, sentence index, and accumulate start and end indices.
    for every token in the line, I want to know its paragraph index, sentence index, token index within the sentence and accumulate start and end indices.
    how should I use it?
    For every spans identified by Chao/Vishnu/both, I will check if there is missed.
    """
    
    # class Token:
    #   def __init__(self, paragraph_index, sentence_index, token_index, accumulate_start_offset, accumulate_end_offset, token_string):
    # class Line:
    #   def __init__(self, paragraph_index, sentence_index, accumulate_start_offset, accumulate_end_offset, sentence_string, tokens):
    output = []
    paragraph_counter = 0
    sentence_counter = 0
    char_offset = 0
    for idx_line, line in enumerate(data):
        if line.strip() == "":
            paragraph_counter += 1
            char_offset += 1
            continue
        
        tmp_line = Line(paragraph_counter, sentence_counter, char_offset, char_offset + len(line), line, path)
        for idx_token, token in enumerate(line.split(" ")):
            tmp_line.tokens.append(Token(paragraph_counter, sentence_counter, idx_token, char_offset, char_offset + len(token.strip()), token.strip()))
            char_offset += len(token.strip()) + 1
        
        sentence_counter += 1
        output.append(tmp_line)
    
    return output

def write_ann_file(ann_content, path):
    entity_counter = 1
    attribute_counter = 1
    
    output = []
    
    for i in ann_content:
        # print(i), type(i)
        output.append("T{}\t{} {} {}\t{}\n".format(entity_counter, i.type, i.start, i.end, i.text))
        if i.elaborated != False:
            output.append("A{}\t{} T{}\n".format(attribute_counter, 'elaborated', entity_counter))
            attribute_counter += 1
        if i.defined != False:
            output.append("A{}\t{} T{}\n".format(attribute_counter, 'defined', entity_counter))
            attribute_counter += 1
        entity_counter += 1
        
    with open(path, "w") as f:
        f.writelines(output)
        
def check_if_two_spans_overlap(span1, span2):
    if span1[1] <= span2[0] or span1[0] >= span2[1]:
        return False
    else:
        return True
    
print(check_if_two_spans_overlap((0, 2), (1, 2)))

def check_if_a_span_is_overlap_with_list_of_spans(span1, spanlist):
    
    if any([check_if_two_spans_overlap(span1, span2) for span2 in spanlist]):
        return True
    else:
        return False
    
print(check_if_a_span_is_overlap_with_list_of_spans((0,1), [(1, 2), (3,5)]))

def judge_the_belongings_of_a_span_in_span_list(span1, span_list):
    start_idx = None
    end_idx = None
    for idx, i in enumerate(span_list):
        if i[0] <= span1[0] < i[1] :
            start_idx = idx
        if i[0] < span1[1] <= i[1]:
            end_idx = idx
            
    return start_idx, end_idx

print(judge_the_belongings_of_a_span_in_span_list((0, 1), [(-1, 0), (0, 1), (3, 5)]))

# parse_txt_file("/nethome/cjiang95/share6/research_18_medical_cwi/data/new-dataset-2023/batch-5-chao-v3/2.txt")

# read_ann_file("/nethome/cjiang95/share6/research_18_medical_cwi/data/new-dataset-2023/batch-5-chao-v3/2.ann")

# parse_txt_file("/nethome/cjiang95/share6/research_18_medical_cwi/data/new-dataset-2023/batch-5-chao-v3/2.txt")[0].tokens


"""
  what do I want to do in this class?
  read in the txt, read in the ann, and merge them to get a doc
  what should a doc class have?
  meta data, complex article, simple article.
  for each line in complex article and simple article, I need to find their associated spans. I some how need to add a entry function / attribute to each line. 
"""

order_for_batch5 = [
    "plos_computational_biology", 
    "plos_genetics", 
    "pnas", 
    "plos_neglected_tropical_diseases", 
    "plos_biology", 
    "cochrane_all", 
    "nhir_Health_Technology_Assessment", 
    "plos_pathogens", 
    "nhir_Public_Health_Research", 
    "nhir_Programme_Grants_for_Applied_Research", 
    "nhir_Health_Services_and_Delivery_Research", 
    "nhir_Efficaacy_and_Mechanism_Evaluation", 
    "MSD", 
    "wiki", 
    "elife",
]

order_for_the_rest = [
    "cochrane_all", 
    "nhir_Efficaacy_and_Mechanism_Evaluation", 
    "nhir_Health_Services_and_Delivery_Research", 
    "nhir_Health_Technology_Assessment", 
    "nhir_Programme_Grants_for_Applied_Research", 
    "nhir_Public_Health_Research", 
    "plos_biology", 
    "plos_computational_biology", 
    "plos_genetics",     
    "plos_neglected_tropical_diseases",    
    "plos_pathogens",     
    "pnas", 
    "MSD", 
    "wiki", 
    "elife",
]

all_batches = [
    "batch-5-chao-v3",
    "batch-7-merge-pass-1",
    "batch-8-mithun",
    "batch-9-mithun",
    "batch-10-mithun",
    "batch-11-mithun",
    "batch-12-mithun",
    "batch-13-mithun",
    "batch-14-mithun",
    "batch-15-mithun",
    "batch-16-mithun",
    "batch-17-mithun",
    ]

class Doc:
  def __init__(self, path):
    
    self.ann_path = path + '.ann'
    self.txt_path = path + '.txt'

    self.source = None
    self.batch = None
    self.article_index = None

    self.lines = parse_txt_file(self.txt_path)
    self.entities = read_ann_file(self.ann_path)

    
    # print(self.entities[0])
    
    # there should be a step to correct the entities, if an entity is not the full string, correct it and its indices
        
    for i in self.entities:
      for j in self.lines:
        if j.check_entity(i) == True:
          # here I need to decide the token_index_start and end for the entity
          # i.token_index_start = ?
          # i.token_index_end = ?
          
          assert i.string in j.string
          # if j.string.count(i.text) > 1:
          # print(i.string)
          # print((i.char_index_start, i.char_index_end))
          # print([[k.accumulate_start_offset, k.accumulate_end_offset] for k in j.tokens])
          
          i.token_index_start, i.token_index_end = judge_the_belongings_of_a_span_in_span_list((i.char_index_start, i.char_index_end), [[k.accumulate_start_offset, k.accumulate_end_offset] for k in j.tokens])
          # else:
          #   # tokenize using split()
          #   tokens_A = i.text.split()
          #   tokens_B = j.string.split()

          #   # find start and end indices
          #   i.token_index_start = tokens_B.index(tokens_A[0])
          #   i.token_index_end = i.token_index_start + len(tokens_A)
            
          j.add_entity(i)
          break
      
    for line in self.lines:
      line.check_cover_full_token()
      
      
    paragraph_buffer = 'meta-data'
    sentence_counter = 0
    
    collect_all_paragraph_indices =sorted(list(set([i.paragraph_index for i in self.lines])))
    assert len(collect_all_paragraph_indices) == 3, "there should be 3 paragraphs in the doc, but observed {} paragraphs".format(len(collect_all_paragraph_indices))
    
    # I need to identify source and the article index based on the path info:
    for idx_batch, batch in enumerate(all_batches):
      if batch in self.txt_path:
        self.batch = batch
        self.article_index = idx_batch
        
    if self.batch == "batch-5-chao-v3":
      for idx in range(15):
        if "{}/{}.txt".format(self.batch, idx) in self.txt_path:
          self.source = order_for_batch5[idx]
    else:
      for idx in range(15):
        if "{}/{}.txt".format(self.batch, idx) in self.txt_path:
          self.source = order_for_the_rest[idx]
      
        
    
    for i in self.lines:
      if i.paragraph_index == min(collect_all_paragraph_indices):
        i.belongings = ['meta-data', 0, self.source, self.batch, self.article_index]
      elif i.paragraph_index == max(collect_all_paragraph_indices):
        i.belongings = ['simple', 0,  self.source, self.batch, self.article_index]
      else:
        i.belongings = ['complex', 0,  self.source, self.batch, self.article_index]
        
      if i.belongings[0] != paragraph_buffer:
        sentence_counter = 0
        paragraph_buffer = i.belongings[0]
      else:
        sentence_counter += 1
        
      i.belongings[1] = sentence_counter
      
def is_sublist(small_list, large_list):
    small_list_len = len(small_list)
    large_list_len = len(large_list)

    output = []
    
    for i in range(large_list_len - small_list_len + 1):
        if large_list[i:i + small_list_len] == small_list:
            output.append(i)
    return output

small_list = [1, 2]
large_list = [0,1,2, 3, 1,2]

result = is_sublist(small_list, large_list)
print(result)  


if __name__ == "__main__":
    
    logger.setLevel(logging.WARNING)    

    # for i in range(15):
    #     for line in Doc("/nethome/cjiang95/share6/research_18_medical_cwi/data/new-dataset-2023/batch-5-chao-v3/{}".format(i)).lines:
    #         line.generate_instance()
    #         # print(line.generate_instance())
    # for i in range(15):
    #     for line in Doc("/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-7-merge-pass-1/{}".format(i)).lines:
    #         line.generate_instance()
    # for i in range(15):
    #     for line in Doc("/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-8-mithun/{}".format(i)).lines:
    #         line.generate_instance()           
    # for i in range(15):
    #     for line in Doc("/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-9-mithun/{}".format(i)).lines:
    #         line.generate_instance()     
    # for i in range(15):
    #     for line in Doc("/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-10-mithun/{}".format(i)).lines:
    #         line.generate_instance()     
    # for i in range(15):
    #     for line in Doc("/coc/pskynet6/cjiang95/research_18_medical_cwi/data/new-dataset-2023/batch-11-mithun/{}".format(i)).lines:
    #         line.generate_instance()   
    # print('done loading')


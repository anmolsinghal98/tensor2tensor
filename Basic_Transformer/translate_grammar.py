# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry


_ENDE_GRAMMAR_CORRECTION_DATASETS=[
  [  
     "content/drive/Shared\ drives/NLP project/data/Basic_Transformer/tmp_dir/train_src_trg.zip",
    ("train.src","train.trg")
  ]
]
_ENDE_EVAL_GRAMMAR_CORRECTION_DATASETS=[
  [
    "content/drive/Shared\ drives/NLP project/data/Basic_Transformer/tmp_dir/valid_src_trg.zip",
    ("dev.src","dev.trg")
  ]
]


@registry.register_problem 
class TranslateGrammarCorrection(translate.TranslateProblem):
  """En-de translation trained on WMT corpus."""

  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_datasets = _ENDE_GRAMMAR_CORRECTION_DATASETS
    return train_datasets if train else _ENDE_EVAL_GRAMMAR_CORRECTION_DATASETS
print('df ')


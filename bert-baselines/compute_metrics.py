# Modified version of Transformers compute metrics script
# Source: https://github.com/huggingface/transformers/blob/v2.7.0/src/transformers/data/metrics/__init__.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, accuracy_score, jaccard_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn
LABEL_TO_TEXT = {"MQ2Q": {0: "غير مكرر", 1: "مكرر"},
                 "FID":  {0: "سخرية", 1: "ليس سخرية"},
                 "OOLD":  {"NOT_OFF": "غير مهين", "OFF": "مهين"},
                 "OHSD": {"NOT_HS": "لا يحض على الكراهية", "HS": "خطاب كراهية"},
                 "XNLI": {"neutral": "علاقة غير مترابطة", "entailment": "علاقة مترابطة", "contradiction": "علاقة متناقضة"},
                 "MDD":  {'SFX': 'صفاقس', 'ALX': 'الإسكندرية', 'ALE': 'حلب', 'FES': 'فاس', 'TRI': 'طرابلس', 'MSA': 'العربية',
                          'CAI': 'القاهرة', 'ASW': 'أسوان', 'AMM': 'عمان', 'TUN': 'تونس', 'DOH': 'الدوحة', 'RIY': 'الرياض',
                          'ALG': 'الجزائر', 'KHA': 'الخرطوم', 'DAM': 'دمشق', 'RAB': 'الرباط', 'SAN': 'صنعاء', 'BEI': 'بيروت',
                          'JER': 'القدس', 'JED': 'جدة', 'BAS': 'البصرة', 'BEN': 'بنغازي', 'SAL': 'سل', 'MUS': 'مسقط',
                          'MOS': 'الموصل', 'BAG': 'بغداد'
                          }
                 }

if _has_sklearn:

    def acc_and_f1(preds, labels, average="binary",pos_label ="entailment"):
        f1 = f1_score(y_true=labels, y_pred=preds, average=average,pos_label= pos_label)
        acc = accuracy_score(preds, labels)
        return {
            "f1": f1,
            "acc": acc,
        }

    def pearson_and_spearman(preds, labels):
        print(preds)
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]

        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    def jaccard_and_f1(preds, labels):
        # print(len(preds[0]))

        jaccard = jaccard_score(y_true=labels, y_pred=preds, average="samples")
        f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
        f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")

        return {
            "jaccard": jaccard,
            "f1-macro": f1_macro,
            "f1-micro": f1_micro,
        }

    def alue_compute_metrics(task_name, preds, labels):
        print(len(preds))
        print(len(labels))
        assert len(preds) == len(labels)
        if task_name == "mq2q":
            return acc_and_f1(preds, labels)
        elif task_name == "mdd":
            return acc_and_f1(preds, labels, average="macro")
        elif task_name == "fid":
            return acc_and_f1(preds, labels,pos_label=1)
        elif task_name == "svreg":
            return pearson_and_spearman(preds, labels)
        elif task_name == "sec":
            return jaccard_and_f1(preds, labels)
        elif task_name == "oold":
            return acc_and_f1(preds, labels,pos_label = LABEL_TO_TEXT["OOLD"]["NOT_OFF"])
        elif task_name == "ohsd":
            return acc_and_f1(preds, labels,pos_label = LABEL_TO_TEXT["OHSD"]["NOT_HS"])
        elif task_name == "xnli":
            return acc_and_f1(preds, labels, average="macro")
        elif task_name =="diag":
            return acc_and_f1(preds, labels, average="macro")
        else:
            raise KeyError(task_name)

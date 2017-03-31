import csv
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score


def read_data(specified_name=None):
    full_tmp_dict = defaultdict(int)

    with open('annotations.csv', newline='') as csvfile:
        read_c_s_v = csv.reader(csvfile, delimiter=',')
        for row in read_c_s_v:
            row[2] = row[2].title()
            if specified_name is None or specified_name == row[0]:
                if row[1] not in full_tmp_dict:
                    full_tmp_dict[row[1]] = {row[2]: 1}
                else:
                    num_dict = full_tmp_dict[row[1]]
                    if row[2] not in num_dict:
                        num_dict[row[2]] = 1
                    else:
                        num_dict[row[2]] += 1
    return full_tmp_dict


def optimise_data(dictionary):
    optimised_dict = defaultdict(int)

    for key in dictionary:
        tmp_dict = dictionary[key]
        max_num = 0
        max_key = ''
        for tmp_key in tmp_dict:
            if tmp_dict[tmp_key] > max_num:
                max_num = tmp_dict[tmp_key]
                max_key = tmp_key
        optimised_dict[key] = max_key
    return optimised_dict

def fleiss_kappa(M):
  """
  See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.

  :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
  :type M: numpy matrix
  """
  np.seterr(divide='ignore', invalid='ignore')
  N, k = M.shape  # N is # of items, k is # of categories
  n_annotators = float(np.sum(M[0, :]))  # # of annotators

  p = np.sum(M, axis=0) / (N * n_annotators)
  P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
  Pbar = np.sum(P) / N
  PbarE = np.sum(p * p)
  print(n_annotators)

  kappa = (Pbar - PbarE) / (1 - PbarE)

  return kappa


full_dict = read_data()
rgen_dict = read_data('rgen8070')

opt_dict = optimise_data(full_dict)
opt_rgen_dict = optimise_data(rgen_dict)

print(opt_dict)

with open('dict.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for key, values in opt_dict.items():
        writer.writerow([key, values, opt_rgen_dict[key]])

industry_name = set()
for key, values in opt_dict.items():
    industry_name.add(values)
for key, values in opt_rgen_dict.items():
    industry_name.add(values)

name_len = len(industry_name)
industry_name = list(industry_name)

A = np.zeros((name_len, name_len))
df = pd.DataFrame(A, index=industry_name, columns=industry_name)

dict_true = np.zeros(len(opt_dict))
dict_rgen = np.zeros(len(opt_dict))

a = 0
for key, values in opt_dict.items():
    values_r = opt_rgen_dict[key]
    i = industry_name.index(values)
    dict_true[a] = i
    j = industry_name.index(values_r)
    dict_rgen[a] = j
    A[i, j] += 1
    a += 1

print(dict_true)
print(dict_rgen)

df.to_csv('df.csv', index=True, header=True, sep='\t')

c_matrix = confusion_matrix(dict_true, dict_rgen)
print(c_matrix)
print(classification_report(dict_true, dict_rgen,
 target_names=industry_name))
print(cohen_kappa_score(dict_true, dict_rgen))

org_names = []
for key in rgen_dict:
    org_names.append(key)
print(org_names)

fless_matrix = np.zeros((150, len(industry_name)))

with open('annotations.csv', newline='') as csvfile:
    read_c_s_v = csv.reader(csvfile, delimiter=',')
    for row in read_c_s_v:
        row[2] = row[2].title()
        rows = org_names.index(row[1])
        columns = industry_name.index(row[2])
        fless_matrix[rows, columns] += 1

print(fless_matrix)
print(fleiss_kappa(fless_matrix))

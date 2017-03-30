import csv
from collections import defaultdict


def read_data(specified_name=None):
    full_tmp_dict = defaultdict(int)

    with open('annotations.csv', newline='') as csvfile:
        read_c_s_v = csv.reader(csvfile, delimiter=',')
        for row in read_c_s_v:
            row[2] = row[2].title()
            print(row)
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

full_dict = read_data()
rgen_dict = read_data('rgen8070')

opt_dict = optimise_data(full_dict)
opt_rgen_dict = optimise_data(rgen_dict)

print(opt_dict)
print(opt_rgen_dict)

industry_name = set()
for key, values in opt_dict.items():
    industry_name.add(values)
print(industry_name)


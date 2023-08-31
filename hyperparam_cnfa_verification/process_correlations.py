import sys
import os

path = "correlations/"

files = os.listdir(path)


losses_dict = {}
accs_dict = {}
init_dict = {}
trained_dict = {}

def add_to_dicts(arch, ps, dict_list):
    for d in dict_list:
        if arch not in d:
            d[arch] = {}

        if ps not in d[arch]:
            d[arch][ps] = []
    return 

for fname in files:
    with open(path + fname, "r") as f:
        lines = f.readlines()
        args = fname.split('_')

        dataset = args[0]
        arch = args[1]
        ps = args[2]

        trained = [float(x) for x in lines[0].split(',')]
        init = [float(x) for x in lines[1].split(',')]
        losses = [float(x) for x in lines[3].split(',')]
        accs = [float(x) for x in lines[5].split(',')]

        add_to_dicts(arch, ps, [losses_dict, accs_dict, init_dict, trained_dict])

        
        losses_dict[arch][ps].append([dataset] + losses)
        init_dict[arch][ps].append([dataset] + init)
        accs_dict[arch][ps].append([dataset] + accs)
        trained_dict[arch][ps].append([dataset] + trained)

def order_dict(out_dict):
    for arch in out_dict.keys():
        for ps in out_dict[arch].keys():
            out_dict[arch][ps] = sorted(out_dict[arch][ps], key=lambda x: x[0])
    
order_dict(losses_dict)
order_dict(init_dict)
order_dict(accs_dict)
order_dict(trained_dict)

import csv

results_path = "results"

for arch in losses_dict.keys():
    for ps in losses_dict[arch].keys():
        with open(f'{results_path}/{arch}_{ps}_losses.csv', "w") as f:
            write = csv.writer(f)
            for row in losses_dict[arch][ps]:
                write.writerow(row)
        
for arch in init_dict.keys():
    for ps in init_dict[arch].keys():
        with open(f'{results_path}/{arch}_{ps}_init.csv', "w") as f:
            write = csv.writer(f)
            for row in init_dict[arch][ps]:
                write.writerow(row)

for arch in accs_dict.keys():
    for ps in accs_dict[arch].keys():
        with open(f'{results_path}/{arch}_{ps}_accs.csv', "w") as f:
            write = csv.writer(f)
            for row in accs_dict[arch][ps]:
                write.writerow(row)


for arch in trained_dict.keys():
    for ps in trained_dict[arch].keys():
        with open(f'{results_path}/{arch}_{ps}_trained.csv', "w") as f:
            write = csv.writer(f)
            for row in trained_dict[arch][ps]:
                write.writerow(row)

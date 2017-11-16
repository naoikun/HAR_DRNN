#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob
import pandas as pd
import numpy as np
import re

#list_onetime_replace
def list_replace(path_list):
    for path_index,file_path in enumerate(path_list):
        path_list[path_index] = path_list[path_index].replace('angless_','*')
    return path_list


#main
dir = '/home/gakusei/kinect/rucksack_right/angle*' #join multi
book_dict = []
path_list = glob.glob(dir)
path_list_replace = list_replace(path_list)

for file_path in path_list_replace:
    jointed_list = []
    for two_path in glob.glob(file_path):
        jointed_list.append(pd.read_csv(two_path))
    df = pd.concat(jointed_list,axis=1)
    df = df.sort_index(axis=1,ascending=True)
    df.to_csv(file_path)


''''
for file_path in glob.glob(dir):
    file_name = os.path.basename(file_path)
    print(file_name)
    book_dict[file_name] = np.loadtxt(file_path,delimiter=',',skiprows=1,usecols=(2,3,4))

print (book_dict)
'''

import numpy as np
import os
import glob
from random import shuffle

data_path = '/home/lupin/SIDD'
files = glob.glob(os.path.join(data_path,'*'))

shutter_speed = {}
for f in files:
    part = f.split('/')
    filename = part[-1].split('_')
    shutter = filename[-3]
    if shutter not in shutter_speed.keys():
        shutter_speed[shutter] = 1
    else:
        shutter_speed[shutter] +=1
        
shutter_top = {}
for i,key in shutter_speed.items():
    if key >=7:
        shutter_top[i] = key
        
iso = [100, 400, 800, 1600, 3200]
shutter_iso = {}
for f in files:
    part = f.split('/')
    filename = part[-1].split('_')
    if (filename[-3] in shutter_top) and (int(filename[-4]) in iso) and ('_'.join([filename[-4],filename[-3]]) not in shutter_iso.keys()):
        shutter_iso['_'.join([filename[-4],filename[-3]])] = 1 
    elif (filename[-3] in shutter_top) and (int(filename[-4]) in iso) and ('_'.join([filename[-4],filename[-3]]) in shutter_iso.keys()):
        shutter_iso['_'.join([filename[-4],filename[-3]])] += 1 

        
iso_class = {}
for val,key in shutter_iso.items():
    temp = val.split('_')
    if temp[0] not in iso_class.keys():
        iso_class[temp[0]] = []
        iso_class[temp[0]].append([val,key])
    else:
        iso_class[temp[0]].append([val,key])

def take_id(files,pattern,num):
    index = []
    for f in files:
        part = f.split('/')
        filename = part[-1].split('_')
        file_pattern = '_'.join([filename[-4],filename[-3]])
        if file_pattern == pattern:
            index.append(int(filename[0]))
    shuffle(index)
    index_part1 = index[:num]
    index_part2 = index[num:]
    return index_part2,index_part1

train_index = []
test_index = []
for i in iso_class.keys():
    for j in iso_class[i]:
        if j[1] >= 4:
            num = int(int(j[1])/3.5)
            temp_train,temp_test = take_id(files,j[0],num)
            train_index += temp_train
            test_index += temp_test

# train_index = [4, 11, 13, 17, 18, 20, 22, 23, 25, 27, 28, 29, 30, 34, 35, 39, 40, 42, 43, 44, 45, 47, 81, 86, 88,
#                      90, 101, 102, 104, 105, 110, 111, 115, 116, 125, 126, 127, 129, 132, 135,
#                      138, 140, 175, 177, 178, 179, 180, 181, 185, 186, 189, 192, 193, 194, 196, 197]
# test_index = [54, 55, 57, 59, 60, 62, 63, 66, 150, 151, 152, 154, 155, 159, 160, 161, 163, 164, 165, 166, 198,
#                      199]
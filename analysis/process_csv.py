import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import re
import os

def create_new_file(file_name):

    csv_item = pd.read_csv('DHF1k_attribute-all.csv')
    attribute_dict = csv_item.to_dict()
    new_string = []
    with open('processed_'  + file_name) as f:
        for line in tqdm.tqdm(f.readlines()):
            video_index = line[13:17]
            new_line = line.strip()
            attributes = [str(key) + ',' + str(attribute_dict[key][int(video_index)-1]) for key in attribute_dict.keys()]
            new_line = new_line + ','.join(attributes) + '\n'
            new_string.append(new_line)
    with open('processed_' + file_name, 'w') as f:
        f.writelines(new_string)
    with open('processed_' + file_name.replace('json','csv'), 'w') as f:
        f.writelines(new_string)
def read_file(file_path):
    new_string = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            new_string.append(line.replace('},', ',\n').replace('{', '').replace('}', '').replace('[', ' ').replace(']', '').replace('\"','').replace(':',',').replace(',,', ','))
    with open('processed_' + file_path, 'w') as f:
        f.writelines(new_string)
    with open('processed_' + file_path.replace('json','csv'), 'w') as f:
        f.writelines(new_string)

def process_file(file_path:str):
    df = pd.read_csv('processed_' + file_path)
    df.columns = [x.strip() for x in df.columns]
    names = list(df.columns[0::2])
    nums = list(df.columns[1::2])[0:len(names)]
    Unname_pattern = re.compile(r'Unnamed: \d+')
    for name in names:
        if Unname_pattern.match(name):
            names.remove(name)
    new_df = pd.DataFrame(np.concatenate((np.array([[float(x) for x in nums]]),np.array(df[nums])),axis=0),columns=names)
    new_df.to_csv(file_path,index=False)

def generate_file(file_path):
    
    read_file(file_path)
    # create_new_file(file_path)
    process_file(file_path.replace('json','csv'))


def main():
    generate_file('sm1-4-600.json')


if __name__ == '__main__':
    main()

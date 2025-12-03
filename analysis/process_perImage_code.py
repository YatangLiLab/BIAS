import os
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_path):
    new_string = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            new_string.append(line.replace('},', ',\n').replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace('\"','').replace(':',',').replace(',,', ','))
    with open('processed_' + file_path, 'w') as f:
        f.writelines(new_string)
    with open('processed_' + file_path.replace('json','csv'), 'w') as f:
        f.writelines(new_string)




def main():
    read_file('MGF-1-1.json')
    read_file('MGF-1-2.json')
    read_file('MGF-1-3.json')
    read_file('MGF-1-5.json')
    read_file('MGF-1-6.json')


if __name__ == '__main__':
    main()

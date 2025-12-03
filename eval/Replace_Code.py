import os
import pandas as pd
import tqdm

def json2csv(filename):
    current_dir = os.getcwd()
    print(current_dir)
    if os.path.exists(filename):
        pass
    else:
        raise FileNotFoundError(f"File {filename} not found")
    with open(filename, 'r') as f:
        lines = f.readlines()
        file_data = {}
        for i in tqdm.trange(len(lines)):
            line = lines[i]
            this_line = line.replace("[{",'')
            this_line = this_line.replace("}]",'')
            this_line = this_line.replace(":",',')
            this_line = this_line.replace(" ",'')
            line_data = this_line.split(',')
            for j in range(len(line_data)//2):
                if file_data.get(line_data[2*j]) is None:
                    if line_data[2*j+1][0] == '\"':
                        file_data[line_data[2*j]] = [line_data[2*j+1]]
                    else:
                        file_data[line_data[2*j]] = [float(line_data[2*j+1])]
                else:
                    if line_data[2*j+1][0] == '\"':
                        file_data[line_data[2*j]].append(line_data[2*j+1])
                    else:
                        file_data[line_data[2*j]].append(float(line_data[2*j+1]))
    data_frame = pd.DataFrame(file_data)
    data_frame.to_csv(filename.replace('.json','.csv'),index=False)

def main(filename):
    json2csv(filename)

if __name__ == '__main__':
        filename = f"target.json"
        main(filename)
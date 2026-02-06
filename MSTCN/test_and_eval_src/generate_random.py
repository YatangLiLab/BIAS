import os
import random
import jsonlines
import copy

tgt_jsonl_lst = ['test_pred/DSTA_label/SM_predictions.jsonl',
                 'test_pred/ustring_label/SM_predictions.jsonl',
                 'test_pred/DRIVE_label/SM_predictions.jsonl']

folder_lst =    ['test_pred/DSTA_label/random',
                 'test_pred/ustring_label/random',
                 'test_pred/DRIVE_label/random']

random_count = 10
# generate 10 example as random examples. 

def generate_random_item(length:int, window_length:int)->list[int]:
    '''
    generate a random item with length, and window_length
    '''
    start_pos = random.randint(0, length - window_length)
    random_item = [(1 if (0<=i-start_pos<window_length) else 0) for i in range(length)]
    return random_item

def tune_random_example(_item:dict[str,any]):
    '''
    _item: dict[str,any] = {
    'predicted': [0/1] sequences of model predicted
    'target': [0/1] sequences of ground truth
    'mask': [1,1,1...,1,0,0....0]-> mask of the sequence
    }
    '''
    total_length = len(_item['predicted'])
    assert total_length == len(_item['target']) == len(_item['mask']), "length of predicted, target, mask should be same."
    assert sum(_item['mask']) == total_length, "mask should be all 1, for valid sequence of generated here"
    window_length = int(sum(_item['predicted'])) # we want to know the window size of generated sequence
    random_item = copy.deepcopy(_item)
    novel_predicted = generate_random_item(total_length, window_length)
    random_item['predicted'] = novel_predicted
    return random_item

def generate_random_lists(jsonl_path:str, target_folder:str ,cnt:int)->None:
    assert os.path.exists(jsonl_path), f"{jsonl_path} not exists."
    assert jsonl_path.endswith('.jsonl'), f"{jsonl_path} should be a jsonl file."
    assert cnt > 0, "cnt should be greater than 0."
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    data = []
    with jsonlines.open(jsonl_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    
    for i in range(cnt):
        random_data = []
        for data_item in data:
            random_data.append(tune_random_example(data_item))
        with jsonlines.open(os.path.join(target_folder, f"SM_random_{i}.jsonl"), 'w') as writer:
            for random_item in random_data:
                writer.write(random_item)


if __name__ == '__main__':
    for jsonl_path, target_folder in zip(tgt_jsonl_lst, folder_lst):
        generate_random_lists(jsonl_path, target_folder, random_count)


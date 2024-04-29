import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('subset_data')
    parser.add_argument('local')
    parser.add_argument('globalC')
    parser.add_argument('result')
    args = parser.parse_args()
    return args

def merge(args):

    with open(args.subset_data) as f:
        data_file = json.load(f)
    
    with open(args.local) as l:
        local_cap = json.load(l)

    with open(args.globalC) as g:
        global_cap = json.load(g)

    for i, data in enumerate(data_file):
        idx = data['instance_id']
        lcd = {}
        for d in local_cap:
            if d['instance_id'] == idx:
                lcd = d 
                break
        gcd = {}
        for d in global_cap:
            if d['instance_id'] == idx:
                gcd = d
                break
        data['local_captions'] = lcd['local_captions'][lcd['inputs']['clue']]
        data['global_captions'] = gcd['global_captions']
        if i % 500 == 0:
            print(f'{i} merging done')
    
    return data_file

def main():
    args = parse_args()
    inferences = merge(args)

    with open(args.result, "w") as f:
        json.dump(inferences, f)

if __name__ == '__main__':
    main()


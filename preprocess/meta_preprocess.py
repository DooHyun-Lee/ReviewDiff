import gzip 
import json 
from tqdm import tqdm 
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/doolee13/ReviewDiff/preprocess/data', type=str)

args = parser.parse_args()

pretrain_cats = ['Automotive']
pretrain_meta_pathes = [f'{args.path}/meta_{cat}.json.gz' for cat in pretrain_cats]
pretrain_seq_pathes= [f'{args.path}/{cat}_5.json.gz' for cat in pretrain_cats]

for path in pretrain_meta_pathes + pretrain_seq_pathes:
    assert os.path.exists(path)

def extract_meta_data(path, meta_data, selected_asins):
    title_len = 0
    total_num = 0
    with gzip.open(path) as f:
        for line in tqdm(f, ncols=100):
            line = json.loads(line)
            attr_dict = dict()
            asin = line['asin']
            if asin not in selected_asins:
                continue

            cat = ''.join(line['category'])
            brand = line['brand']
            title = line['title']

            title_len += len(title.split())
            total_num += 1

            attr_dict['title'] = title
            attr_dict['brand'] = brand
            attr_dict['category'] = cat
            meta_data[asin] = attr_dict
    return title_len, total_num


if __name__ == '__main__':
    meta_asins = set()
    seq_asins = set()


    for path in pretrain_meta_pathes:
        with gzip.open(path) as f:
            for line in tqdm(f):
                line = json.loads(line)
                if line['asin'] is not None and line['title'] is not None:
                    meta_asins.add(line['asin'])

    for path in pretrain_seq_pathes:
        with gzip.open(path) as f:
            for line in f:
                line = json.loads(line)
                if line['asin'] is not None and line['reviewerID'] is not None:
                    seq_asins.add(line['asin'])

    # asin and title in meta file
    # asin, reviewerID, summary in 5-core file
    selected_asins = meta_asins & seq_asins
    print(f'Meta has {len(meta_asins)} Asins.')
    print(f'Seq has {len(seq_asins)} Asins.')
    print(f'{len(selected_asins)} Asins are selected.')

    meta_data = dict()
    for path in pretrain_meta_pathes:
        extract_meta_data(path, meta_data, selected_asins)

    with open('meta_data.json', 'w', encoding='utf8') as f:
        json.dump(meta_data, f)
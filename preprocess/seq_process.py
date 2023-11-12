import gzip 
import json
from tqdm import tqdm 
import os
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='/home/doolee13/ReviewDiff/preprocess/data', type=str)

args = parser.parse_args()

pretrain_cats = ['Automotive']
pretrain_seq_pathes= [f'{args.path}/{cat}_5.json.gz' for cat in pretrain_cats]

for path in pretrain_seq_pathes:
    assert os.path.exists(path)

# dictionary with key : asin // val : attributes
meta_data = json.load(open('meta_data.json'))

# key : userId, val : interaction log
train_seqs = defaultdict(list)
val_seqs = defaultdict(list)
test_seqs = defaultdict(list)

def meta_to_sentence(asin):
    attr_dict = meta_data[asin]
    title = attr_dict['title']
    brand = attr_dict['brand'] 
    cat = attr_dict['category']
    sentence = f"This product, titled '{title}' and branded as {brand}, falls under the category of {cat}."
    return sentence

sequences = defaultdict(list)
miss_cnt = 0
valid_cnt = 0

with gzip.open(pretrain_seq_pathes[0]) as f:
    for line in tqdm(f):
        line = json.loads(line)
        userid = line['reviewerID']
        asin = line['asin']
        time = line['unixReviewTime']
        if asin in meta_data and line.get('overall', None) is not None:
            review = line['overall']
            temp_dict = {}
            temp_dict['attribute'] = meta_to_sentence(asin)
            temp_dict['review'] = review
            temp_dict['asin'] = asin
            sequences[userid].append((time,temp_dict))
            valid_cnt += 1
        else:
            miss_cnt += 1

length = 0
training_data = []
for user, sequence in tqdm(sequences.items()):
    sequences[user] = [ele[1] for ele in sorted(sequence, key=lambda x: x[0])]
    training_data.append(sequences[user])
    length += len(sequences[user])

print(f'Averaged length : {length/len(sequences)}')

with open('train_data.json', 'w') as f:
    json.dump(training_data, f)
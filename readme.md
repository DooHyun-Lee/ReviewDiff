# Preprocessing steps
1) download and place {category}_5_.json.gz and meta_{category}.json.gz file in preprocess/data directory. 
    - you could download file [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

2) run python meta_preprocess.py with --path argument(path to your preprocess/data/ directory)
    - this will create meta_data.json file (dictionary with key: item_id and value: item attributes)

3) run python seq_process.py with --path argument(path to your preprocess/data directory)
    - this will create train_data.json file
    - each line of json file : List[dict] (contain each item in {dict} format in sequential order)
    - dictionary keys
      - attribute(item expressed in a sentence form)
      - review(numpy type review between 1~5)
      - item id(asin)

# Data folder structure
├── embeddings
│   └── distilbert
│       └── embeddings_768.pckl
├── preprocessed
│   ├── meta_data.json
│   └── train_data.json
└── raw
    ├── Automotive_5.json.gz
    └── meta_Automotive.json.gz

# Preprocessed data format
**train_data** is a list where each entry is a trajectory, which is a list of dicts each representing a product.
```python
[
    {'attribute': "This product, titled 'Hardley Dangerous Illusions Sizzling Flame Sticker BURNT ORANGE' and branded as Hardley Dangerous Illusions, falls under the category of AutomotiveExterior AccessoriesBumper Stickers, Decals & Magnets.",
    'review': 5.0,
    'asin': 'B000VZGTPY'
    }, ...
]
```
It has type `List[List[dict]]]` and length 193651.

**meta_data** is a dictionary mapping `asin` to a dictionary of metadata.
Here's an example with the key "B000VZGTPY":
```python
{
    'title': 'Hardley Dangerous Illusions Sizzling Flame Sticker BURNT ORANGE',
    'brand': 'Hardley Dangerous Illusions',
    'category': 'AutomotiveExterior AccessoriesBumper Stickers, Decals & Magnets'
}
```
It has type `Dict[str, dict]` and contains 79317 keys.

# Embeddings
The embeddings are stored in a pickle file, organized as a list of trajectories, where each trajectory is a list of dictionaries corresponding to products. Each dictionary in a trajectory contains the following keys:
- embedding: A NumPy array representing the sentence embedding of the product's attributes. The embeddings are generated using the DistilBERT model and currently have a dimension of 768.
- review: The review rating of the product.
- asin: The Amazon Standard Identification Number (ASIN) of the product.

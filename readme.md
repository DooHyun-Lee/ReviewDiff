## Preprocess steps
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
import fire
import datasets
from datasets import load_dataset
dataset = load_dataset("Hantao/ChemReactionImageRE", cache_dir='/home/dijinli/Disk/Workspace/multi-modal-relation-extraction/data/hf')
def data_from(k, pair, limit):
    ocr = k['ocr-token']
    # pair_sets = map(set, k['relation-pair'])
    if pair in k['relation-pair']:
        class_label = 1
        e1_idx = ocr.index(list(pair)[0])
        e2_idx = ocr.index(list(pair)[1])

    else:
        if limit > 70:
            return None
        class_label = 0
        e1_idx = ocr.index(list(pair)[0])
        e2_idx = ocr.index(list(pair)[1])
    k['pair'] = pair
    k['class-label'] = class_label
    k['e1_e2'] = [e1_idx, e2_idx]
    
    return k
def generate_ds():
    for k in dataset['train']:
        ocr_list = k['ocr-token']
        cnt = 0
        for ocr1 in ocr_list:
            for ocr2 in ocr_list:
                if ocr1 == ocr2:
                    continue
                tmp = data_from(k,[ocr1, ocr2], cnt)
                if tmp:
                    yield tmp
                cnt = cnt+1
        
from datasets import Dataset

def main(
    output_dir: str = './output_data',
    input_dir: str = './input_data',
    
):
    ds = Dataset.from_generator(generate_ds)
    ds.save_to_disk('/home/dijinli/Disk/Workspace/multi-modal-relation-extraction/data/processed')
    
if __name__ == '__main__':
  fire.Fire(main)
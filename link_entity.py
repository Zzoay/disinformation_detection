
import os
import json
import operator
import time
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import tagme
from retrying import retry
from wikidata.datavalue import DatavalueError
from wikidata.client import Client


# your tagme token
tagme.GCUBE_TOKEN = ""


def read_text(data_file):
    try:
        with open(data_file, 'r') as f:
            line = f.readline()
            dct = json.loads(line)
            return dct['text']
    except FileNotFoundError:
        return


def record_files_gen(data_path):
    """
    -gossipcop
    -politifact
    -real
    -fake
        -politifact31
            -news content.json
            -tweets
        -...
    """
    for label in os.listdir(data_path):
        records_path = data_path + "/" + label
        for record in os.listdir(records_path):
            record_file = records_path + "/" + record + "/news content.json"
            yield label, record_file


def build_vocab(data_path):
    word_vocab = {}
    for data in record_files_gen(data_path):
        text = read_text(data[1])
        if text is None:
            continue
        for word in text.split():
            try:
                word_vocab[word] += 1
            except KeyError:
                word_vocab[word] = 1
    word_vocab = sorted(word_vocab.items(), key=operator.itemgetter(1), reverse=True)
    return word_vocab


def count_len(data_path):
    lens = []
    cnt = {'real':0, 'fake':0}
    for data in record_files_gen(data_path):
        label = data[0]
        text = read_text(data[1])
        if text is None or len(text.split()) == 0:
            continue
        cnt[label] += 1  

        lens.append(len(text.split()))
    
    print(sum(lens) / len(lens))
    print(max(lens))
    print(min(lens))
    print(cnt)


def done(future, *args,**kwargs):
    # print(future,args,kwargs)
    future.result()#get the response object
    # print(response)


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def retry_ann(text):
    return tagme.annotate(text)


def extract_entities(data_path, client, break_point=None, stop=None):
    pool=ThreadPoolExecutor(5000)
    max_len = 512
    # data_dct = {'real': [], 'fake':[]}
    cnt = 1
    for data in record_files_gen(data_path):
        data_item = {}

        if break_point is not None and cnt <= break_point:
            cnt += 1
            continue
        if stop is not None and cnt > stop:
            break
        
        label = data[0]
        text = read_text(data[1])

        if text is None or len(text.split()) == 0:
            continue

        word_lst = text.split() 
        if len(word_lst) > max_len:
            text = " ".join(word_lst[:max_len])

        data_item["text"] = text
        data_item["entities"] = []

        if (cnt+1) % 300 == 0:
            time.sleep(90)

        with open(f'data/politifact/{label}/{cnt}.txt', 'w+', encoding='utf-8') as f:
            f.write(text.replace('\n', ' ') + '\n')
        
        lunch_annotations = retry_ann(text)
        if lunch_annotations is None:
            continue
        for i, ann in enumerate(lunch_annotations.get_annotations(0.2)):  # type: ignore
            # entity = process_one(ann, client)
            v = pool.submit(process_one, *(ann, client, label, cnt))
            v.add_done_callback(done)
            # entity = v.result()
            # print(entity)

        cnt += 1
    print(cnt)


def process_one(ann, client, label, cnt):
    entity = {"entity_name": '', "index": [], "neighbours": []}

    idx = [ann.begin, ann.end]
    title = ann.entity_title

    entity["entity_name"] = title
    entity["index"] = idx

    url = f'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&format=json&titles={quote(title)}'
    try:
        r = urlopen(url)
    except URLError:
        write_entity(entity, label, cnt)
        return
    pages = json.load(r)
    r.close()
    page, = pages['query']['pages'].values()
    try:
        wiki_id = page['pageprops']['wikibase_item']
        getter = client.get(wiki_id, load=True)
    except (KeyError, URLError):
        write_entity(entity, label, cnt)
        return
    
    try:
        for e in getter.iterlists():
            if len(entity["neighbours"]) >= 10:  # set max num of neighbour as 10
                break
            r = e[0]
            tmp = e[1]
            
            try:
                for x in tmp:
                    entity["neighbours"].append(str(x.label))
            except AttributeError:
                break
    except DatavalueError: 
        write_entity(entity, label, cnt)
        return
    
    print(entity)
    write_entity(entity, label, cnt)


def write_entity(entity, label, cnt, corpus='politifact'):
    if entity is not None:
        with open(f'data/{corpus}/{label}/{cnt}.txt', 'a+', encoding='utf-8') as f:
            f.write(str(entity) + '\n')


def extract_entities_wo_neighbours(data_path, corpus, break_point=None):
    cnt = 0
    for data in record_files_gen(data_path):
        label = data[0]
        text = read_text(data[1])

        if break_point is not None and cnt <= break_point:
            cnt += 1
            continue
        
        if text is None or len(text.split()) == 0:
            continue

        with open(f'data/{corpus}/{label}/{cnt}.txt', 'w+', encoding='utf-8') as f:
            f.write(text.replace('\n', ' ') + '\n') 
        
        lunch_annotations = retry_ann(text)
        if lunch_annotations is None:
            continue
        for i, ann in enumerate(lunch_annotations.get_annotations(0.1)):  # type: ignore
            entity = {"entity_name": '', "index": []}
            
            idx = [ann.begin, ann.end]
            title = ann.entity_title

            entity["entity_name"] = title
            entity["index"] = idx

            write_entity(entity, label, cnt, corpus)

        cnt += 1


if __name__ == "__main__":
    corpus = "gossipcop"
    # time.sleep(18000)

    data_dir = r"\FakeNewsNet\code\fakenewsnet_dataset"
    data_path = data_dir + r"/" + corpus
    # client = Client()

    # extract_entities(data_path, client, break_point=9980, stop=10000)
    # p = multiprocessing.Pool()

    extract_entities_wo_neighbours(data_path, corpus)

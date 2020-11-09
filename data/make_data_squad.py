from collections import Counter
import csv
import re
import json
def load(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            data.append(line)
    data.pop(0)
    return data


def to_squad(data, context, outfile):
    res = {"version": "v1.0",
           "data": []}
    for d in data:
        c = re.sub('\s+','',context[d[1]])
        a = re.sub('\s+','',d[3])
        q = re.sub('\s+','',d[2])
        start = c.find(a)
        if start == -1:
            continue
        tmp = {"paragraphs": [{"id": d[1], "context": c,
                              "qas": [{
                                  "question": q,
                                  "id": d[1],
                                  "answers": [{
                                      "text": a,
                                      "answer_start":start}]
                              }]}],
               "id": d[0],
               "title": d[2][:3]}
        res['data'].append(tmp)
    print('get', len(res['data']))
    with open(outfile, 'w') as f:
        json.dump(res, f, ensure_ascii=False)

context = load('passage.csv')
context = {k[0]: k[1] for k in context}
train = load('train.csv')

to_squad(train, context, 'train_squad.json')
to_squad(train[:1000], context, 'dev_squad.json')

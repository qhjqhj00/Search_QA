import json
import numpy as np
import csv
from collections import defaultdict, Counter

def load(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            data.append(line)
    head = data.pop(0)
    return data

def positify(x):
    m = min(x)  - 1
    for i,_ in enumerate(x):
        x[i] = x[i] - m
    return x

test = load('../data/NCPPolicies_test.csv')
test = {k[0]:k[1] for k in test}

data=json.loads(open('../output/nbest_test.json').read())

rank_c = Counter()
norm = defaultdict(list)

for k in data:
    for d in data[k]:
        if d['text'] != 'empty' and len(d['text']) > 1:
            ids = k.split('_')[0]
            d['rank'] = int(k.split('_')[1])
            d['score'] = d['start_logit']+d['end_logit']
            norm[ids].append(d)

for k in norm:
    norm[k] = sorted(norm[k], key=lambda x: x['score'], reverse=True)
    tmp = {'rank':[], 'case':[]}
    for case in norm[k]:
        if case['rank'] in tmp['rank']:
            continue
        else:
            tmp['rank'].append(case['rank'])
            tmp['case'].append(case)
    norm[k] = tmp['case']


for k in norm:
    score_list = []
    for case in norm[k]:
        score_list.append(case['score'])
    score_list = positify(score_list)
    for i,case in enumerate(norm[k]):
        norm[k][i]['score'] = score_list[i]   
        if norm[k][i]['rank'] == 0:
            norm[k][i]['score'] *= 1
        elif 1<=norm[k][i]['rank']<3:
            norm[k][i]['score'] *= 0.9
        elif 3<=norm[k][i]['rank']<5:
            norm[k][i]['score'] *= 0.85
        elif 5<=norm[k][i]['rank']<10:
            norm[k][i]['score'] *= 0.8
        elif 15> norm[k][i]['rank'] >=10:
            norm[k][i]['score'] *= 0.75
        elif norm[k][i]['rank'] >=15:
            norm[k][i]['score'] *= 0.7
    norm[k] = sorted(norm[k], key=lambda x: x['score'], reverse=True)
    rank_c[norm[k][0]['rank']] += 1


for k in norm:
    if int(norm[k][0]['rank']) >19:
        print(k)
        print(test[k])
        print(norm[k][0]['rank'])
        print(norm[k][0]['text'])
        print(norm[k][0]['score'])
        print('*'*12)

top1 = sum([rank_c[k] for k in range(1)]) / 1643
top3 = sum([rank_c[k] for k in range(3)]) / 1643
top5 = sum([rank_c[k] for k in range(5)]) / 1643
top10 = sum([rank_c[k] for k in range(10)]) / 1643
top20 = sum([rank_c[k] for k in range(20)]) / 1643

print(f'top1:\t{top1}')
print(f'top3:\t{top3}')
print(f'top5:\t{top5}')
print(f'top10:\t{top10}')
print(f'top20:\t{top20}')

with open('../output/submit.csv','w') as f:
    f.write('id\tdocid\tanswer\n')
    for k in norm:
        t = norm[k][0]['text']
        f.write(f'{k}\t{k}\t{t}\n')
    f.truncate()

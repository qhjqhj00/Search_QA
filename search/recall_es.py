import sys
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import re
import json
import jieba
import jieba.posseg as pseg
import jieba.analyse
import csv
from tqdm import tqdm
from split import split_text
from pydict import Dict
from collections import Counter

RE_TOPICS = set([
    r'"(.+?)"',
    r'“(.+?)”',
    r'<(.+?)>',
    r'《(.+?)》',
    r'\[(.+?)\]',
    r'【(.+?)】',
    r'〖(.+?)〗',
    r'#(.+?)#',
    r'『(.+?)』',
    r'「(.+?)」',
    r'“(.+?)"',
])

def get_topics(text):
    topics = []
    for r in RE_TOPICS:
        topics += re.findall(r, text)
    return topics

def pasteWord(itemList, posList, newPos):
    outList = []
    nLex = []
    for termList in itemList:
        term = termList[0]
        pos = termList[1]
        if pos in posList:
            nLex.append(term)
        else:
            outList.append(termList)
            if len(nLex) > 0:
                outList.append(["".join(nLex), newPos])
                nLex = []
    if len(nLex) > 0:
        outList.append(["".join(nLex), newPos])
    return outList

def mergeLexN(lineList):
    itemList = []
    for term in lineList[1].split(")"):
        termList = term.replace("(","").split(",")
        if len(termList) != 2:
            continue
        termList[1] = termList[1].replace('"', '').strip()
        itemList.append(termList)
    itemList = pasteWord(itemList, ["n", "nz", "vn", "nw"], "n")
    itemList = pasteWord(itemList, ["v", "vd"], "v")
    return itemList

def delete_common(l):
    dont = ['疫情','新冠']
    l = [w for w in l if w not in dont]
    
def getTagMap():
    queryDict = {}
    f = open("../data/processed_comb.txt")  # 该文件的描述和示例请看../NLU 目录和../data 目录
    datalines = f.readlines()
    f.close()
    for line in datalines:
        line = line.strip()
        lineList = line.split("\t")
        outLineList = mergeLexN(lineList)
        ents = []
        nouns = []
        for itemList in outLineList:
            pos = itemList[1]
            if pos  in ("LOC", "ORG"):
                if len(itemList[0]) < 2:
                    continue
                ents.append(itemList[0])
            elif pos in ("n", "nz", "vn", "nw"):
                    if len(itemList[0]) > 3:
                        nouns.append(itemList[0])
        topics = get_topics(lineList[0])
        delete_common(nouns)
        delete_common(ents)
        delete_common(topics)
        queryDict[lineList[0]] = {'n': nouns, 'e': ents, 't':topics}
    return queryDict

def load(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            data.append(line)
    data.pop(0)
    return data

def analyseQuery(queryDict, query):
    if query in queryDict:
        return queryDict[query]
    else:
        return {'n': [], 'e': [], 't':[]}

es = Elasticsearch(
    ['192.168.1.29'],
    port=9200
)

queryDict = getTagMap()
data = load('../data/NCPPolicies_test.csv') 

di = Dict("data")

jres = {"version": "v1.0",
        "data": []}
for d in tqdm(data):
    query = analyseQuery(queryDict, d[-1])
    q = re.sub('\s+','',d[-1])
    tagList = jieba.analyse.extract_tags(d[-1], topK = 8)
    tagList = list(set(tagList))
    delete_common(tagList)
    passage_match = di.multi_match(d[-2])
    passage_code = list({passage_match[k]['value']['code'][:2] for k in passage_match})
    q_l = list(set(query['e'] + query['t'] + query['n']))
    doc = {
            "query": {
            "function_score": {
                "query":{
                "dis_max": {
                "queries": [
                    { "match": { "passage": {"query":re.sub('\s+','',d[-1]),"boost":2.5}}},
                    { "match": { "passage": {"query":','.join(tagList),"boost":2.5}}},
                    { "match": { "passage": {"query":','.join(q_l),"boost":2.5}}},
                    { "match": { "word_phrase": {"query":','.join(tagList),"boost":1.5}}},
                    { "match": { "entities": {"query":','.join(q_l),"boost":1}}},
                    { "match": { "ad": {"query":' '.join(passage_code),"boost":8}}}
                ],
                    "tie_breaker": 0.4
                }},
                "functions": [],
                "score_mode": "sum",
                "boost_mode": "sum"}
        }}
    
    for e in query['e']:
        doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": e}},"weight": 4})
    for t in query['t']:
        doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 6})
    for t in query['n']:
        doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 5})
    for t in tagList:
        doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 8})
    # print(doc)
    
    results = es.search(index='0408', doc_type='_doc', body=doc, size=30)['hits']['hits']
    context = ''
    for i,res in enumerate(results):
            context = res['_source']['passage']
            context_id = res['_id']
            maxlen = 512 - len(q) - 10
            sub_text, _ = split_text(context, maxlen)
            for m, sub in enumerate(sub_text):
            
                tmp = {"paragraphs": [{"id": context_id, "context": sub,
                                      "qas": [{
                                          "question": q,
                                          "id":f'{d[0]}_{i}_{m}',
                                          "answers": [{
                                              "text": '',
                                              "answer_start":0}]
                                      }]}],
                       "id": f'{d[0]}_{i}_{m}',
                       "title": q[:3]}
                jres['data'].append(tmp)

with open('../data/test_squad_30.json', 'w') as f:
    json.dump(jres, f, ensure_ascii=False, indent=4)


                


import sys
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import re
import json
import jieba
import jieba.posseg as pseg
import jieba.analyse
import csv

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
    f = open("./processed_comb.txt")
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

es = Elasticsearch(
    ['192.168.1.29'],
    port=9200
)

def cut(s):
    return pseg.cut(s,use_paddle=True)
keep = set(['PER','ORG','LOC','vn','n','v'])
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


def analyseQuery(queryDict, query):
    if query in queryDict:
        return queryDict[query]
    else:
        return {'n': [], 'e': [], 't':[]}

queryDict = getTagMap()
data = load('NCPPolicies_test.csv')
cc = load('NCPPolicies_context_20200301.csv')
cc = {k[0]:k[1] for k in cc}
from tqdm import tqdm
n = 0
r = 0
from collections import Counter
top = Counter()

with open('NCPPolicies_test_recall.csv', 'w') as f:
    jres = {"version": "v1.0",
           "data": []}
    for d in tqdm(data):
        n += 1
        query = analyseQuery(queryDict, d[-1])
        q = re.sub('\s+','',d[-1])
        tagList = jieba.analyse.extract_tags(d[-1], topK = 8)
        tagList = list(set(tagList))
        delete_common(tagList)
        #if len(query) == 0:
        q_l = list(set(query['e'] + query['t'] + query['n']))
        doc = {
              "query": {
                "function_score": {
                    "query":{
                  "dis_max": {
                    "queries": [
                        { "match": { "passage": {"query":q,"boost":2.5}}},
                        { "match": { "word_phrase": {"query":','.join(tagList),"boost":1.5}}},
                        { "match": { "entities": {"query":','.join(q_l),"boost":1}}}
                    ],
                      "tie_breaker": 0.35
                  }},
                  "functions": [],
                  "score_mode": "sum",
                  "boost_mode": "multiply"}
            }}
        
        for e in query['e']:
            doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": e}},"weight": 4})
        for t in query['t']:
            doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 6})
        for t in query['n']:
            doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 5})
        for t in tagList:
            doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 8})
        print(doc)
        
        results = es.search(index='0408', doc_type='_doc', body=doc, size=5)['hits']['hits']
        context = ''
        for i,res in enumerate(results):
            context += res['_source']['passage']
            context_id = res['_id']
        f.write(f'{d[0]}\t{context_id}\t{context}\t{d[-1]}\n')
        tmp = {"paragraphs": [{"id": context_id, "context": context,
                              "qas": [{
                                  "question": q,
                                  "id":d[0],
                                  "answers": [{
                                      "text": '',
                                      "answer_start":0}]
                              }]}],
               "id": d[0],
               "title": q[:3]}
        jres['data'].append(tmp)
    with open('test_squad.json', 'w') as f:
        json.dump(jres, f, ensure_ascii=False, indent=4)


                


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
    f = open("./queryWord")
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

def p(s):
    s = s.replace("/", "")
    topics = get_topics(s)
    if len(topics) > 0:
        return " AND ".join([json.dumps(x,ensure_ascii=False) for x in topics])
    ret = set()
    for w, t in cut(s):
        #print(w,t)
        if t in keep and len(w)>1:
            ret.add(w)
    if len(ret) == 0:
        return s
    else:
        return "(%s)"%" AND ".join(["(%s)"%x if len(x)>3 else json.dumps(x, ensure_ascii=False) for x in ret])+" OR (%s)"%s
        #return "%s"%" AND ".join(["(%s)"%x if len(x)>3 else json.dumps(x, ensure_ascii=False) for x in ret])

def esSearch(word):
    #doc = {
    #  "query": {
    #    "match": {
    #      "text": word
    #    }
    #  }
    #}
    doc = {
      "query": word
    }
    ret = es.search(index='base_content_202020330', doc_type='news_info', body=doc)
    return ret

queryDict = getTagMap()
data = load('NCPPolicies_train_20200301.csv')
cc = load('NCPPolicies_context_20200301.csv')
cc = {k[0]:k[1] for k in cc}
from tqdm import tqdm
n = 0
r = 0
from collections import Counter
top = Counter()

with open('NCPPolicies_train_recall.csv', 'w') as f:
    bad = []
    for d in tqdm(data):
        n += 1
        #query = d[-2]
        query = analyseQuery(queryDict, d[-2])
        #if len(query) == 0: query.append('')
        #query = query.replace('"', '').replace("/", "").replace("[","").replace("]","")
        #query = query.split(' OR ')[0]
        #tagList = jieba.analyse.extract_tags(d[-2], topK = 3)
        tagList = jieba.analyse.extract_tags(d[-2], topK = 8)
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
                        { "match": { "passage": {"query":re.sub('\s+','',d[-2]),"boost":2.5}}},
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
        
        """doc['query']['function_score']['functions'].append(
                                {"field_value_factor": {
                                "field": "word_phrase",
                                "modifier": "log1p",
                                "factor": 0.1}})"""
        print(doc)
        
        #print(doc)
        """{"query": {
    "dis_max": {
        "queries": [
            { "match": { "passage": {"query":d[-2],"boost":2}}},
            { "match": { "word_phrase": {"query":','.join(tagList),"boost":1.2}}},
            { "match": { "entities": {"query":','.join(query),"boost":1}}}
                ],"tie_breaker": 0.3
            }
        }}"""
        """else:
            doc = {
              "query": {
                "function_score": {
                    "query":{
                      "dis_max": {
                            "queries": [
                                { "match": { "passage": {"query":d[-2],"boost":1.25}}},
                                { "match": { "word_phrase": ','.join(tagList)}}
                                    ],"tie_breaker": 0.25
                                }},
                  "functions": [
                    {
                      "filter": {
                        "term": {
                          "passage": query[0]
                        }
                      },
                      "weight": 1
                    }
                  ],
                  "score_mode": "sum",
                  "boost_mode": "multiply"
                }
              }
            }"""
            
        """{"query": {
        "dis_max": {
            "queries": [
                { "match": { "passage": {"query":d[-2],"boost":1.25}}},
                { "match": { "word_phrase": ','.join(tagList)}}
                    ],"tie_breaker": 0.25
                }
            }}"""
        
            
        """doc = {
                  "query": {
                    "dis_max": {
                      "queries": [
                        {
                          "match": {
                            "passage": {
                              "query": d[-2],
                              "minimum_should_match": "50%",

                            }
                          },
                          "match": {
                            "word_phrase": {
                              "query": d[-2],
                              "minimum_should_match": "50%"
                            }
                          }}
                        ],
                        "tie_breaker": 0.3
                      }
                    }
                  }"""
        results = es.search(index='0408', doc_type='_doc', body=doc, size=5)['hits']['hits']
        for i,res in enumerate(results):
            context = res['_source']['passage']
            if i == 0: top1 = context
            context_id = res['_id']
            if context.find(re.sub('\s+','',d[-1])) != -1:
                top[i] += 1
                r +=1
                f.write(f'{d[0]}\t{context_id}\t{context}\t{d[-2]}\t{d[-1]}\n')
                break
            if i == 2:
                bad.append({'query':d[-2],'right':cc[d[1]], 'top-1':top1, 'answer': d[-1], 'es': doc})
print(len(bad))
with open('bad_case.json', 'w') as f:
    json.dump(bad,f,indent=4,ensure_ascii = False)
                
print(r/n)
print(top)


#def checkAnswer(query):
#    results = es.search(index='base_content_202020330', doc_type='news_info', q=query)['hits']['hits']
#    flag = -1
#    for i,res in enumerate(results):
#        context = res['_source']['text']
#        context_id = res['_id']
#        if context.find(d[-1]) != -1:
#            flag = 1
#            return True
#    return False
#
#with open('NCPPolicies_train_recall.csv', 'w') as f:
#    for d in tqdm(data):
#        n += 1
#        query = analyseQuery(queryDict, d[-2])
#        query = query.replace('"', '').replace("/", "").replace("[","").replace("]","")
#        flag = checkAnswer(query)
#        if not flag:
#            print(query)

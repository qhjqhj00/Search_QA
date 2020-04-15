#encoding:utf-8
import time
import jieba
import jieba.analyse
import sys
import re
from elasticsearch import Elasticsearch
from elasticsearch import helpers


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

def getTagMap():
    queryDict = {}
    f = open("./queryWord")
    datalines = f.readlines()
    f.close()
    for line in datalines:
        line = line.strip()
        lineList = line.split("\t")
        outLineList = mergeLexN(lineList)
        k_item = []
        i_item = []
        for itemList in outLineList:
            pos = itemList[1]
            if pos  in ("LOC", "ORG"):
                if len(itemList[0]) < 2:
                    continue
                k_item.append(itemList[0])
        topics = get_topics(lineList[0])
        queryDict[lineList[0]] = k_item+topics
    return queryDict

class ElasticObj:
    def __init__(self, index_name, index_type, ip ="192.168.1.29"):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        '''
        self.index_name =index_name
        self.index_type = index_type
        self.es = Elasticsearch([ip])


    def create_index(self,index_name,index_type):
        '''
        创建索引,创建索引名称为ott，类型为ott_type的索引
        :param ex: Elasticsearch对象
        :return:
        '''
        #创建映射
        _index_mappings ={
                       "mappings":{
                          "properties":{
                               "word_phrase": {
                                  "type": "text",
                                  "analyzer": "ik_smart"
                                },
                             "entities": {
                                  "type": "text",
                                  "analyzer": "ik_smart"
                                },
                             "passage":{
                                "type":"text",
                                "analyzer":"ik_max_word",
                                "search_analyzer":"ik_smart"
                             },
                             "docid":{
                                "type":"text"
                             }
                          }
                       }
        }
        if self.es.indices.exists(index=self.index_name) is not True:
            res = self.es.indices.create(index=self.index_name, body=_index_mappings)
            print(res)

    def bulk_Index_Data(self):
        ACTIONS = []
        i = 1
        #d = getTagMap()
        with open("context_with_kw.csv") as f:
            for line in f:
                if i == 1:
                    i += 1
                    continue
                lineList = line.strip().split("\t")
                doc_id = lineList[0]
                content = lineList[1]
                tagList = jieba.analyse.extract_tags(content, topK = 10)
                #lineList[2] = lineList[2].replace(',',' ')
                if len(lineList) == 2:
                    lineList.append('')
                action = {
                    "_id": doc_id,
                    "_index": self.index_name,
                    "_type": self.index_type,
                    "_source": {
                        "passage": re.sub('\s+','',content),
                        "docid": doc_id,
                        "entities": lineList[2],
                        "word_phrase": ','.join(tagList)
                    }
                }
                i += 1
                ACTIONS.append(action)
        print(len(ACTIONS))
        success, _ = helpers.bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)

    def Get_Data_By_Body(self, word):
        doc = {
            "query": {
                "match": {
                    "passage": word
                }
            },
            "from": 0,
            "size": 10
        }
        _searched = self.es.search(index=self.index_name, doc_type=self.index_type, body=doc)
        return _searched


if __name__ == '__main__':
    obj = ElasticObj("0408", "_doc", "192.168.1.29")
    obj.create_index("0408", "_doc")
    obj.bulk_Index_Data()

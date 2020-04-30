#encoding:utf-8
import time
import jieba
import jieba.analyse
import sys
import re
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from pydict import Dict

class ElasticObj:
    def __init__(self, index_name, index_type, passage_path, ip ="192.168.1.29"):
        '''
        :param index_name: 索引名称
        :param index_type: 索引类型
        :passage_path: 文章路径
        '''
        self.index_name =index_name
        self.index_type = index_type
        self.passage_path = passage_path
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
                             "ad":{
                                "type":"text",
                                "analyzer":"whitespace",
                                "search_analyzer":"whitespace"
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
        d = Dict("data")
        with open(self.passage_path) as f: 
            for line in f:
                if i == 1:
                    i += 1
                    continue
                lineList = line.strip().split("\t")
                doc_id = lineList[0]
                content = lineList[1]
                tagList = jieba.analyse.extract_tags(content, topK = 10)
                passage_match = d.multi_match(content)
                passage_code = list({passage_match[k]['value']['code'][:2] for k in passage_match})
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
                        "word_phrase": ','.join(tagList),
                        "ad":' '.join(passage_code)
                    }
                }
                i += 1
                ACTIONS.append(action)
        print(len(ACTIONS))
        success, _ = helpers.bulk(self.es, ACTIONS, index=self.index_name, raise_on_error=True)
        print('Performed %d actions' % success)


if __name__ == '__main__':
    passage_path = '../data/context_with_kw.csv'
    ip = "192.168.1.29"
    index_name = "0408"
    index_type = "_doc"
    obj = ElasticObj(index_name, index_type, ip=ip, passage_path=passage_path)
    obj.create_index(index_name, index_type)
    obj.bulk_Index_Data()

from flask import Flask, request, Response
import json
import sys
import os
from pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from tools import utils
import jieba
from preprocess.cmrc2018_output import write_predictions
from preprocess.cmrc2018_preprocess import json2features

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--init_restore_dir', type=str, required=True)
args = parser.parse_args()

print('init model...')
if 'albert' not in args.init_restore_dir:
    model = BertForQuestionAnswering(bert_config)
else:
    if 'google' in args.init_restore_dir:
        model = AlbertForMRC(bert_config)
    else:
        model = ALBertForQA(bert_config, dropout_rate=args.dropout)
utils.torch_show_all_params(model)
utils.torch_init_model(model, args.init_restore_dir)

def get_context(query):
    tagList = jieba.analyse.extract_tags(query, topK = 8)
    doc = {
        "query": {
        "function_score": {
            "query":{
            "dis_max": {
            "queries": [
                { "match": { "passage": {"query":q,"boost":2.5}}},
                { "match": { "word_phrase": {"query":','.join(tagList),"boost":1.5}}},
            ],
                "tie_breaker": 0.35
            }},
            "functions": [],
            "score_mode": "sum",
            "boost_mode": "multiply"}
    }}
        
    for t in tagList:
        doc['query']['function_score']['functions'].append({"filter": {"term":{"passage": t}},"weight": 8})
    results = es.search(index='0408', doc_type='_doc', body=doc, size=3)['hits']['hits']
    context = []
    for i,res in enumerate(results):
        context.append(res['_source']['passage'])
    return context

def tokenizer(query, context):
    pass

def to_feature(query,context):
    pass

def to_result():
    pass

@app.route('/',methods=['GET','POST'])
    if request.method == 'POST':
        query = request.form["query"]
    else:
        query = request.args["query"]
    context = get_context(query)
    res = to_result()
    result = {"result": res}
    
    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=4510,threaded=True)
    



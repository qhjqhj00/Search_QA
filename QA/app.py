from flask import Flask, request, Response
import json
import sys
import os
from pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from tools import utils
from tqdm import tqdm
import jieba
import argparse
import collections
from elasticsearch import Elasticsearch
from preprocess.cmrc2018_output import write_predictions
from preprocess.cmrc2018_preprocess import json2features
from tools import official_tokenization as tokenization
import torch
from torch.utils.data import TensorDataset, DataLoader

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--init_restore_dir', type=str, required=True)
parser.add_argument('--bert_config_file', type=str, required=True)
parser.add_argument('--vocab_file', type=str, required=True)
args = parser.parse_args()
SPIECE_UNDERLINE = '▁'
print('init model...')
bert_config = args.bert_config_file
model = BertForQuestionAnswering(bert_config)
utils.torch_show_all_params(model)
utils.torch_init_model(model, args.init_restore_dir)
es = Elasticsearch(
    ['192.168.1.29'],
    port=9200
)
def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def get_context(query):
    tagList = jieba.analyse.extract_tags(query, topK = 8)
    doc = {
        "query": {
        "function_score": {
            "query":{
            "dis_max": {
            "queries": [
                { "match": { "passage": {"query":q,"boost":2.5}}},
                { "match": { "passage": {"query":','.join(tagList),"boost":2.5}}},
                { "match": { "word_phrase": {"query":','.join(tagList),"boost":1.5}}},
            ],
                "tie_breaker": 0.4
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

def json2features(query: str, content: list, tokenizer, is_training=False, repeat_limit=3, max_query_length=96,
                  max_seq_length=512, doc_stride=128):
    train_data = content
    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    def _is_chinese_char(cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def is_fuhao(c):
        if c == '。' or c == '，' or c == '！' or c == '？' or c == '；' or c == '、' or c == '：' or c == '（' or c == '）' \
                or c == '－' or c == '~' or c == '「' or c == '《' or c == '》' or c == ',' or c == '」' or c == '"' or c == '“' or c == '”' \
                or c == '$' or c == '『' or c == '』' or c == '—' or c == ';' or c == '。' or c == '(' or c == ')' or c == '-' or c == '～' or c == '。' \
                or c == '‘' or c == '’':
            return True
        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp) or is_fuhao(char):
                if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                    output.append(SPIECE_UNDERLINE)
                output.append(char)
                output.append(SPIECE_UNDERLINE)
            else:
                output.append(char)
        return "".join(output)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
            return True
        return False

    # to examples
    examples = []
    mis_match = 0
    for i,context in tqdm(enumerate(train_data)):
        context_chs = _tokenize_chinese_chars(context)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in context_chs:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            if c != SPIECE_UNDERLINE:
                char_to_word_offset.append(len(doc_tokens) - 1)

            ques_text = query
            ans_text = ''

            start_position_final = None
            end_position_final = None

            examples.append({'doc_tokens': doc_tokens,
                                'orig_answer_text': ans_text,
                                'qid': i,
                                'question': ques_text,
                                'answer': ans_text,
                                'start_position': start_position_final,
                                'end_position': end_position_final})


    # to features
    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example['question'])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example['doc_tokens']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            features.append({'unique_id': unique_id,
                             'example_index': example_index,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': start_position,
                             'end_position': end_position})
            unique_id += 1

    return examples, features

def test(model, eval_examples, eval_features, device):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = './predictions.json'
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=16, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=5, max_answer_length=500,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

def to_feature(query,context):
    pass

def to_result():
    pass

@app.route('/',methods=['GET','POST'])
def model_1():
    if request.method == 'POST':
        query = request.form["query"]
    else:
        query = request.args["query"]
    context = get_context(query)
    examples, features = json2features(query, context)
    test(model, examples, features, '0')
    
    return Response(json.dumps({'result'}, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=4510,threaded=True)
    



from flask import Flask, request, Response
import json
from pytorch_modeling import BertConfig, BertForQuestionAnswering, ALBertConfig, ALBertForQA
from tools import utils
from tqdm import tqdm
import jieba
import argparse
import collections
from elasticsearch import Elasticsearch
from tools import official_tokenization as tokenization
import torch
from torch.utils.data import TensorDataset, DataLoader
import jieba.analyse
import math
from data_utils import json2features, get_predictions, get_context

app = Flask(__name__, static_url_path='')

parser = argparse.ArgumentParser()
parser.add_argument('--init_restore_dir', type=str, required=True)
parser.add_argument('--bert_config_file', type=str, required=True)
parser.add_argument('--vocab_file', type=str, required=True)
args = parser.parse_args()

print('init model...')
bert_config = BertConfig.from_json_file(args.bert_config_file)
model = BertForQuestionAnswering(bert_config)
tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
model.cuda()
utils.torch_show_all_params(model)
utils.torch_init_model(model, args.init_restore_dir)

es = Elasticsearch(
    ['192.168.1.29'],
    port=9200
)

def predict(model, eval_examples, eval_features):
    device = torch.device("cuda")
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
    #print(all_results)
    all_predictions = get_predictions(eval_examples, eval_features, all_results,
                      n_best_size=5, max_answer_length=500,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)
    return all_predictions


@app.route("/api")
def model_1():
    query = request.args.get('text', '')
    #if request.method == 'POST':
    #    query = request.form["query"]
    #else:
    #    query = request.args["query"]
    print(query)
    context = get_context(query, es)
    print(context)
    examples, features = json2features(query, context, tokenizer)

    res = predict(model, examples, features)
    
    return Response(json.dumps({'status':"ok", 'message':res, 'query':query},ensure_ascii=False), mimetype="application/json")

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4633, debug=True)
    





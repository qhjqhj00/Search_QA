python app.py \
--init_restore_dir ../output/robert/model.pth \ # 在这修改模型路径
--bert_config_file bert_config.json \ # 在这修改模型配置路径
--vocab_file vocab.txt \
--port 4633 \
--es_ip 192.168.1.29 \
--es_port 9200
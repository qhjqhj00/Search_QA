# 检索介绍

### 检索部分的效果

​	本方案检索阶段基于ElasticSearch，使用了开源的词法标注工具对query和passage进行了一些语义理解。训练数据集5000条，我们评测检索召回率的方法是在top-k文章中，匹配到答案就算成功。召回率测试结果如下：

| Top-k:  | 1     | 3     | 5     | 10    | 20    | 30    | 50    |
| ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 召回率: | 0.775 | 0.881 | 0.911 | 0.942 | 0.963 | 0.974 | 0.982 |

​	权衡召回率和模型预测时间，本方案选择top-30的文章进行预测。

### 对query和passage的理解

### 	![search](../data/images/search.png)



​	本方案使用Paddle开源词法分析模型和jieba的关键词模块进行查询和文档的理解，对于分析结果筛选部分，分配权重。

https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis

​	由于这次比赛是政务文档，大部分内容都有具体的行政区划归属，所以对query和passage都进行了省级行政区划提取。

下面是一个文档的示例。其中，"entities" 是使用paddle的词法分析工具得到；"word_phrase"是使用jieba.analyse得到；"ad" 是取文中出现行政区划的代码前两位（省级）。

```shell
{
  "passage":"南安：举报违规餐饮店奖励口罩50个昨日，《南安市防控新型冠状病毒感染的肺炎疫情应急指挥部第17号通告》发布，进一步规范餐饮经营活动。根据通告，餐饮服务所有从业人员一律佩戴口罩上岗，从事直接入口食品岗位的还应佩戴手套。全面推行打包和外卖服务，一律采取“无接触”方式进行，原则上外卖送餐员不得进入店内取餐。严禁堂食、严禁包厢服务、严禁任何形式的聚餐活动。严禁街边烧烤、大排档等夜市餐饮经营活动。餐饮行业夜间经营时段不得超过21时。对不履行规定、拒不整改的一律关停整顿。对围桌聚餐、聚众饮酒等违规聚集的，一律关停，对相关责任单位和个人由公安机关按规定予以严肃处理。违规单位（个人）列入重点监管对象，造成严重後果的，列入失信黑名单，并予以曝光。为鼓励社会监督，对查证属实的举报，奖励举报人一次性口罩50个。举报电话：12315、110。",
  "docid":"b85afb2d621937799f35368f03da2f77",
  "entities":"南安市,南安",
  "word_phrase":"严禁,一律,口罩,餐饮,外卖,50,违规,聚餐,关停,通告",
  "ad":"35"
}
```

下面是一个query的示例，分析方法与passage一样：

```shell
{
  "query":{
    "function_score":{
      "query":{
        "dis_max":{
          "queries":[
            {"match":{"passage":{"query":"内江市近期去过武汉的人出现感染症状如何处理？","boost":2.5}}},
            {"match":{"passage":{"query":"如何,症状,内江市,武汉,处理,感染,近期,去过","boost": 2.5}}},
            {"match":{"passage":{"query":"感染症状,内江市,武汉","boost":2.5}}},
            {"match":{"word_phrase":{"query":"如何,症状,内江市,武汉,处理,感染,近期,去过","boost":1.5}}},
            {"match":{"entities":{"query":"感染症状,内江市,武汉","boost":1}}},
            {"match":{"ad":{"query":"42 51","boost":8}}}
          ],
          "tie_breaker":0.4
        }
      },
      "functions":[
        {"filter":{"term":{"passage":"内江市"}},"weight":4},
        {"filter":{"term":{"passage":"武汉"}},"weight":4},
        {"filter":{"term":{"passage":"感染症状"}},"weight":5},
        {"filter":{"term":{"passage":"如何"}},"weight":8},
        {"filter":{"term":{"passage":"症状"}},"weight":8},
        {"filter":{"term":{"passage":"内江市"}},"weight":8},
        {"filter":{"term":{"passage":"武汉"}}, "weight":8},
        {"filter":{"term":{"passage":"处理"}}, "weight":8},
        {"filter":{"term":{"passage":"感染"}}, "weight":8},
        {"filter":{"term":{"passage":"近期"}}, "weight":8},
        {"filter":{"term":{"passage":"去过"}}, "weight":8}
      ],
      "score_mode": "sum",
      "boost_mode": "sum"
    }
  }
}
```

### ik自定义词典

​	ES分词准确程度非常重要，本方案使用了ik自定义词典 (427665个词) ，其中包括了所有的passage和query的语义理解结果，对于单字进行了过滤。同时也加入了一个自定义的停用词词典 (767个词）。
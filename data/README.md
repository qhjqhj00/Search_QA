本方案使用疫情政务问答助手比赛数据，详情请参考比赛页面：
https://www.datafountain.cn/competitions/424/datasets



使用paddle词法分析工具后的文件保存格式如下：

```
在北京市法律援助中心服务大厅开始对外提供法律咨询之前，如何获得法律咨询服务？\t(在, p)(北京市法律援助中心服务大厅, ORG)(开始, v)(对外, vd)(提供, v)(法律, n)(咨询, vn)(之前, f)(,, w)(如何, r)(获得, v)(法律, n)(咨询, vn)(服务, vn)(?, w)
```
可以修改run.sh 中 infering部分的输入文件路径，然后运行。（详细步骤请参考lexical_analysis/README.md)
```
cd lexical_analysis
sh run.sh infering
```


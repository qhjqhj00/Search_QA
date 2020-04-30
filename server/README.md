### 本方案还附加赠送一个Web服务

修改 app.sh 中的模型文件路径，配置文件路径和vocab路径，端口和es地址即可部署：

```shell
sh app.sh
```

部署之后页面长这样，它仅使用top-1的结果进行预测：
![serve](./serve.png)
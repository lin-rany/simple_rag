### 概述
### 快速上手
```
pip install -r requirements.txt
```
llm.py ChatUseLangChainSDKV6中使用的为本地跑的Embedding模型，可使用以下命令下载
```
pip install -U huggingface_hub
huggingface-cli download --resume-download BAAI/bge-large-zh --local-dir BAAI/bge-large-zh
```
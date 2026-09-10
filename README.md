# LLM学习项目


## 整体思路
本项目主要用于LLM学习
1. 了解Transformer Decoder的架构和实现
2. 针对Decoder各个组件最前沿的变种，进行学习和实现
3. 重点学习各种前沿的注意力机制和MOE架构，尝试做一点算子融合
4. 实现简单的训练infra，完成预训练、SFT、Agentic RL
5. 实现简单的模型评估
6. 学习LLM的目的，是为了知道推理引擎的算子要怎么写，不太追求模型的性能和效果，差不多就行了，重点放在推理上

为了方便学习，分词器、训练数据，就都统一使用openbmb/MiniCPM5-1B
选择这个模型没有很solid的理由，它效果比较好、文档比较好，我没有做太充分的调研

## Tokenizer
https://huggingface.co/openbmb/MiniCPM5-1B/tree/main
直接复用这里的tokenizer
我自己尝试写了bbpm分词器，感觉逻辑比较简单，对推理的学习帮助不是很大，所以重构项目的时候我就把这部分给删掉了，直接所有模型都复用同一个tokenizer

## 上下文长度
上下文长度统一8192

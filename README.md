# Synthesize, Diagnose, and Optimize: Towards Fine-Grained Vision-Language Understanding [![arXiv](https://img.shields.io/badge/arXiv-2312.00081-b31b1b.svg)](https://arxiv.org/abs/2312.00081)
Wujian Peng , Sicheng Xie, Zuyao You, [Shiyi Lan](https://voidrank.github.io/), [Zuxuan Wu](https://zxwu.azurewebsites.net/)



## Abstract
> Vision language models (VLM) have demonstrated remarkable performance across various downstream tasks.
> However, understanding fine-grained visual-linguistic concepts, such as attributes and inter-object relationships,
> remains a significant challenge. While several benchmarks aim to evaluate VLMs in finer granularity, their primary 
> focus remains on the linguistic aspect, neglecting the visual dimension. Here, we highlight the importance of
> evaluating VLMs from both a textual and visual perspective. We introduce a progressive pipeline to synthesize
> images that vary in a specific attribute while ensuring consistency in all other aspects. Utilizing this data engine,
> we carefully design a benchmark, SPEC, to diagnose the comprehension of object size, position, existence, and count.
> Subsequently, we conduct a thorough evaluation of four leading VLMs on SPEC. Surprisingly, their performance is close to random guess, 
> revealing significant limitations. With this in mind, we propose a simply yet effective approach to optimize VLMs
> in fine-grained understanding, achieving significant improvements on SPEC without compromising the zero-shot performance. 
> Results on two additional fine-grained benchmarks also show consistent improvements, further validating the transferability
> of our approach.

## Overview of SPEC Benchmark
> SPEC consists of six distinct subsets, distributed across the dimensions of Size, Position, Existence and Count.
> Each test case consists of an image candidate set, which differs only in certain visual concept, and a text candidate set, 
> which differs only in corresponding language concept. Due to space constraints, we present a maximum of three images and texts here, 
> however, more comprehensive test cases are available in the supplementary material.
<p align="center">
<img src="figs/spec_overview.png" width="1080px"/>  
<br>
</p>

## TODO
- [ ] Release the demo of our data synthesis pipeline
- [ ] Release the data of SPEC benchmark
- [ ] Release the evaluation code of SPEC


****
If you find our work useful, please consider citing it:

```
@article{SPEC,
  title={Synthesize, Diagnose, and Optimize: Towards Fine-Grained Vision-Language Understanding},
  author={Wujian Peng, Sicheng Xie, Zuyao You, Shiyi Lan, Zuxuan Wu}, 
  journal={arXiv preprint arXiv:2312.00081},
  year={2023}
}
```

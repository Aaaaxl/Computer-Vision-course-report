# 计算机视觉课程实验汇总

本仓库收录了我们在《计算机视觉》课程汇报中制作 PPT 过程中所使用的实验代码与结果，主要包含以下三部分：

1. **DeepLabV3+ 上的任务模型比较**  
2. **UniLSeg 自动标注引擎复现**  
3. **UniLSeg 模型分割性能评估**

---

## 重要声明：

本仓库并没有对任何模型做改进工作，只是单纯地使用 DeepLabV3+、SAM、BLIP、CLIP、UniLSeg 模型，并加载了作者训练好的权重，做了一些效果呈现的可视化工作。如果你想查看本仓库的内容，请点开各文件夹下的 jupyter 文件即可，如果你想详细了解各模型的源码，我们在此贴上了各源码的链接：
[DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch)
[SAM](https://github.com/facebookresearch/segment-anything)
[BLIP](https://github.com/salesforce/BLIP)
[CLIP](https://github.com/openai/CLIP)
[UniLSeg](https://github.com/yongliu20/UniLSeg)

## 1. DeepLabV3+ 任务模型比较

我们基于 [DeepLabV3+] 框架，分别选择了两种主干网络以及数据集进行对比实验：

- **MobileNetV2 (Cityscapes) &ensp;——&ensp;OS=16**  
  实验基于 [MobileNet]，在 [Cityscapes] 数据集上进行训练与测试，主要考察轻量化模型在城市场景分割任务中的表现。

- **ResNet50 (VOC) &ensp;——&ensp;OS=16**  
  实验基于 [ResNet-50]，在 [PASCAL VOC 2012] 数据集上进行训练与测试，侧重评估中等规模网络在通用目标分割任务中的性能。

> **结果对比**  
> - 训练集外（跨域/其它任务）泛化性能  
> - 模型参数量与推理速度  

---

## 2. UniLSeg 自动标注引擎复现

在本部分，我们尝试复现 [UniLSeg] 中的自动标注（automatic annotation engine）模块：

- 仅展示 **Box-labeled** 数据的标注处理流程  
- 包含数据预处理、标注生成脚本及可视化示例  

---

## 3. UniLSeg 模型分割性能评估

最后，使用预训练的 UniLSeg 模型，在不同任务（如语义分割、实例分割等）上进行评估，并展示可视化结果。

---

## 运行环境

- Python == 3.10  
- PyTorch == 2.1.0  
- CUDA == 12.1

---

## 引用
  
[UniLSeg: 《Universal Segmentation at Arbitrary Granularity with Language Instruction》](https://arxiv.org/abs/2312.01623)

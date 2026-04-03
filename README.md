方言语音识别模型项目说明

项目简介
本项目聚焦中文多方言语音识别任务，包含数据切分、预处理、模型训练、解码评估与可视化演示等完整流程。
项目在开源基座模型上进行微调与结构增强，支持贪心解码、Beam 解码与 Beam 加语言模型解码。

主要功能
一、数据准备与切分，生成训练集、验证集与测试集。
二、离线预处理与数据增强，提升训练效率与鲁棒性。
三、方言增强模型训练，支持断点续训与多指标评估。
四、KenLM 融合解码评估，支持参数搜索与分方言分析。
五、本地 Web 演示服务，便于展示与验证识别效果。

快速开始
1. 运行 fen.py 生成 split_data 下的数据切分文件。
2. 按需运行 preprocess_data.py 进行离线预处理。
3. 运行 dialect_fine_tune.py 进行训练。
4. 运行 dialect_evaluate.py 或 run_lm_eval.py 进行评估。
5. 如需构建语言模型，运行 tools/build_kenlm.py。

开源协议说明

模型与项目代码
本项目的微调模型、训练与推理代码，基于 Apache License 2.0 协议开源。
本项目基于采用 Apache License 2.0 协议的开源基座模型进行微调，衍生作品完整继承原协议条款，完整协议内容见仓库根目录 LICENSE 文件。
基座模型地址如下。
https://huggingface.co/wbbbbb/wav2vec2%2Dlarge%2Dchinese%2Dzh%2Dcn
本项目在此基础上进行微调和增强。

训练数据集
本项目训练所用数据集版权归 Magic Data Technology Co., Ltd. 所有，基于 Creative Commons Attribution－NonCommercial 4.0 International License（CC BY－NC 4.0）协议提供。
⚠️ 该数据集仅限非商业用途使用，二次使用、分发需遵守原协议要求，保留原版权方署名，禁止任何商业性使用。

使用建议
请在使用前确认依赖环境、模型文件与数据许可范围。
如需对外发布模型或服务，请先完成合规审查并遵守对应协议条款。

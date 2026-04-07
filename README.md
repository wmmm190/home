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

模型发布（Hugging Face）
本项目的 full 版本模型已公开发布在 Hugging Face：
https://huggingface.co/HF19030626674/Hdialect_wav2vec2
该仓库对应本地训练产物中的 dialect_model_best_full，可用于直接推理或二次微调。

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
参考文献
[1] S.-H. Wang, Z.-C. Chen, J. Shi, M.-T. Chuang, G.-T. Lin, K.-P. Huang, D. Harwath, S.-W. Li, and H.-yi. Lee, “How to Learn a New Language? An Efficient Solution for Self-Supervised Learning Models Unseen Languages Adaption in Low-Resource Scenario,” arXiv preprint. arXiv:2411.18217, 2025. [Online]. Available: https://arxiv.org/abs/2411.18217.
[2] W. Liu, J. Hou, D. Yang, M. Cao, and T. Lee, “A Parameter-efficient Language Extension Framework for Multilingual ASR,” in Proc. Interspeech 2024, 2024, pp. 3929–3933. doi: 10.21437/Interspeech.2024-1745.
[3] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented Transformer for Speech Recognition,” in Proc. Interspeech 2020, 2020, pp. 5036–5040. doi: 10.21437/Interspeech.2020-3015.
[4] R. Feng, Y.-L. Liu, Z.-H. Ling, and J.-H. Yuan, “Wav2f0: Exploring the Potential of Wav2vec 2.0 for Speech Fundamental Frequency Extraction,” in 2024 IEEE 14th International Symposium on Chinese Spoken Language Processing (ISCSLP), Beijing, China, 2024, pp. 169–173. doi: 10.1109/ISCSLP63861.2024.10800188.
[5] A. Pasad, J.-C. Chou, and K. Livescu, “Layer-Wise Analysis of a Self-Supervised Speech Representation Model,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), Cartagena, Colombia, 2021, pp. 914–921. doi: 10.1109/ASRU51503.2021.9688093.
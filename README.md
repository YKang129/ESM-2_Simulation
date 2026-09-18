# ESM-2 蛋白语言模型：药物靶点表征与相似性分析

计算生物学与人工智能药学作业。本项目用 Meta 的 **ESM-2 蛋白语言模型**（Protein
Language Model, pLM）对 16 个知名药物靶点蛋白生成嵌入向量，并做**相似性分析、
层次聚类、降维可视化**，从「蛋白序列相似 → 潜在功能/药物敏感性相似」的角度展示
AI 在药物研发中的一种典型应用。

## 项目背景

传统药物研发靠实验逐个筛选靶点，成本高、周期长。**蛋白质语言模型**借鉴了 NLP
（自然语言处理）里"单词 → 句子"的思想，把 20 种氨基酸当作"字母"、整条蛋白序列
当作"句子"，在大规模蛋白序列上做自监督预训练。训练好的模型能把任意一条蛋白序列
编码成一个固定长度的**向量（embedding）**，编码相近的蛋白往往在结构/功能上也相近。

本项目用 ESM-2 对一组抗癌药物的靶点蛋白（EGFR、KRAS、BRAF 等）编码，验证：
**同属一个蛋白家族（尤其都是激酶）的靶点，其嵌入向量应当更相似、在聚类图上更靠近。**

## 数据来源

16 条蛋白序列均来自 [UniProt](https://www.uniprot.org/) 的**真实标准序列**
（`data/drug_targets.fasta`），蛋白名、UniProt 编号、家族分类与代表药物见
`data/drug_targets_metadata.csv`。覆盖 5 个家族：

| 家族 | 代表靶点 | 代表药物 |
|------|----------|----------|
| 受体酪氨酸激酶 (RTK) | EGFR、ERBB2、KDR、ALK、FLT3 | 吉非替尼、曲妥珠单抗、舒尼替尼 |
| 非受体酪氨酸激酶 | ABL1、SRC、BTK、JAK2 | 伊马替尼、达沙替尼、伊布替尼 |
| 丝/苏氨酸激酶 | BRAF、MAPK1、CDK4、CDK6 | 维莫非尼、帕博西尼 |
| 小 GTPase | KRAS | 索托拉西布 |
| 其他酶类 | ACE2、PARP1 | 奥拉帕尼等 |

## 目录结构

```
esm2_drug_targets/
├── README.md                  # 本说明
├── requirements.txt           # Python 依赖
├── analysis.ipynb             # 主分析 notebook（加载数据→嵌入→聚类→可视化）
├── data/
│   ├── drug_targets.fasta     # 16 条药物靶点蛋白序列（UniProt）
│   └── drug_targets_metadata.csv
├── src/
│   ├── __init__.py
│   └── embed_targets.py       # 可复用工具库 + 命令行脚本
└── results/                   # 运行后生成的图表与嵌入（.npz 不入库）
```

## 环境与安装

- Python 3.10+（本项目在 **Python 3.14** 下验证通过）
- 本项目默认模型 `esm2_t6_8M` 极小，**CPU 即可运行，无需 GPU**

### 方式 A（推荐）：独立虚拟环境 + CPU 版 PyTorch

CPU 版 torch 无需 GPU、内存占用小（约 300MB），在任何机器上都能复现，
也不会影响你已安装的 GPU 版 torch。**内存有限（可用内存 < 4GB）时务必用这种方式**，
因为 GPU 版 torch 光 `import` 就要加载约 3GB 的 CUDA 库：

```bash
cd esm2_drug_targets

# 1) 创建虚拟环境（--system-site-packages 复用系统已装包，只需补装 CPU torch）
python -m venv --system-site-packages .venv

# 2) 安装 CPU 版 torch（版本与 GPU 版相同，只是不含 CUDA）
.venv\Scripts\python -m pip install --no-deps --force-reinstall "torch==2.13.0+cpu" ^
    --index-url https://download.pytorch.org/whl/cpu

# 3) 启动 notebook
.venv\Scripts\python -m jupyter notebook analysis.ipynb
```

> 若系统里还没有 numpy/pandas/matplotlib 等基础依赖，先执行
> `.venv\Scripts\python -m pip install -r requirements.txt` 再装 CPU torch。

### 方式 B：直接安装全部依赖（GPU 版 torch + 内存充足时）

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

> 国内下载 HuggingFace 模型：代码已自动设置 `HF_ENDPOINT=https://hf-mirror.com`，
> 无需科学上网。

## 复现步骤

**方式一（推荐）：直接运行 Jupyter notebook**

```bash
jupyter notebook analysis.ipynb
```

按顺序执行所有单元格即可，图表会内嵌在 notebook 中并同时保存到 `results/`。

**方式二：命令行生成嵌入**

```bash
python src/embed_targets.py --fasta data/drug_targets.fasta \
    --model facebook/esm2_t6_8M_UR50D --out results/embeddings.npz
```

## 模型选择

| 模型 | 参数量 | 下载大小 | 嵌入维度 | 说明 |
|------|--------|----------|----------|------|
| `esm2_t6_8M_UR50D` | 0.08 亿 | ~32 MB | 320 | 默认，最轻量、可复现 |
| `esm2_t30_150M_UR50D` | 1.5 亿 | ~600 MB | 640 | 质量更好，平衡之选 |
| `esm2_t33_650M_UR50D` | 6.5 亿 | ~2.6 GB | 1280 | 质量最高，需 ~8GB 以上可用内存 |

在 `analysis.ipynb` 或 `src/embed_targets.py` 中修改 `MODEL_NAME` 即可切换。
模型越大嵌入质量越高，但越占内存、下载越慢。

## 方法简述

1. **编码**：ESM-2 对每条序列做前向传播，得到每个残基的隐向量，再**平均池化**成
   一个蛋白级向量（长序列切窗后平均，避免截断）。
2. **相似度**：对向量做 L2 归一化后求内积，即**余弦相似度**。
3. **聚类**：对相似度矩阵做层次聚类（`scipy` 的 `linkage`），画带树状图的**热图**。
4. **降维**：用 PCA / t-SNE 把高维向量投影到二维，按蛋白家族着色观察分组。

## 主要结论

- 激酶类靶点（受体/非受体酪氨酸激酶、丝苏氨酸激酶）在嵌入空间中明显聚在一起；
- 非激酶靶点（ACE2、PARP1、KRAS）与激酶显著分开；
- 说明 ESM-2 在**无监督**（没给任何标签）情况下，仅靠蛋白序列就学到了"蛋白家族"
  这一生物学结构，可作为药物靶点发现、脱靶风险分析的基础。

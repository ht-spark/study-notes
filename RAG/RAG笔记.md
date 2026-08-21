## **RAG学习笔记**

## 一、RAG的基本概念

### 1. 什么是RAG？解决的核心问题是什么？

	RAG（Retrieval-Augmented Generation）是一种旨在解决大语言模型知识陈旧和模型幻觉问题的技术范式。它的核心是将模型内部学到的“**参数化知识**”（模型权重中固化的、模糊的“记忆”），与来自外部知识库的“**非参数化知识**”（精准、可随时更新的外部数据）相结合。其运作逻辑就是在 LLM 生成文本前，先通过检索机制从外部知识库中动态获取相关信息，并将这些“参考资料”融入生成过程，从而提升输出的准确性和时效性。

### 2.RAG的基础技术原理

RAG主要分为：检索和生成两个阶段

**(1) 检索阶段----寻找非参数化知识**

- **知识向量化**：**嵌入模型（Embedding Model）** 将外部知识库编码为**向量索引（Index）**，存入**向量数据库**。
- **语义召回**：当用户发起查询时，检索模块利用同样的嵌入模型将**问题向量化**，并通过**相似度搜索（Similarity Search）**，从海量数据中精准锁定与问题最相关的文档片段。

**(2) 生成阶段----如何非参数化知识和参数化知识**

- **上下文整合**：**生成模块**接收到检索阶段送来的相关文档片段以及用户的原始问题。
- **指令引导生成**：该模块会遵循预设的 **Prompt** 指令，将上下文与问题有效整合，并引导 LLM进行可控的、有理有据的文本生成。

### 3.RAG的工作流程

```
上传文档 -> 文档清洗与增强 -> 文档chunk -> 向量化嵌入(vector+payload) -> 

查询理解与向量化 -> 检索 -> 构建提示词(检索到的文本内容+系统提示词+用户问题) -> LLM生成回答
```

## 二、数据准备

**数据准备主要完成:文档上传加载、文档清洗与增强、文档chunk工作。**

### 1.文档加载

#### 1.1 作用是什么？

	文档加载器负责将各种格式的非结构化文档（PDF、Word等）转换为程序可以处理的结构化文档，如：Markdown等。

#### 1.2 要做什么事？

（1）解析不同格式的原始文档，将 PDF、Word等内容提取为可处理的结构化文档，如：Markdown等。

（2）在解析过程中同时抽取文档来源、页码、作者等关键信息作为**元数据(metadata)。**

（3）把文本和元数据整理成统一的数据结构，方便后续进行切分、向量化入库。

#### 1.3 当前主流RAG文档加载器

<div align="center">
<table border="1" style="margin: 0 auto;">
  <tr>
    <th style="text-align: center;">工具名称</th>
    <th style="text-align: center;">特点</th>
    <th style="text-align: center;">适用场景</th>
    <th style="text-align: center;">性能表现</th>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>PyMuPDF4LLM</strong></td>
    <td style="text-align: center;">PDF→Markdown转换，OCR+表格识别</td>
    <td style="text-align: center;">科研文献、技术手册</td>
    <td style="text-align: center;">开源免费，GPU加速</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>TextLoader</strong></td>
    <td style="text-align: center;">基础文本文件加载</td>
    <td style="text-align: center;">纯文本处理</td>
    <td style="text-align: center;">轻量高效</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>DirectoryLoader</strong></td>
    <td style="text-align: center;">批量目录文件处理</td>
    <td style="text-align: center;">混合格式文档库</td>
    <td style="text-align: center;">支持多格式扩展</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Unstructured</strong></td>
    <td style="text-align: center;">多格式文档解析</td>
    <td style="text-align: center;">PDF、Word、HTML等</td>
    <td style="text-align: center;">统一接口，智能解析</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>FireCrawlLoader</strong></td>
    <td style="text-align: center;">网页内容抓取</td>
    <td style="text-align: center;">在线文档、新闻</td>
    <td style="text-align: center;">实时内容获取</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>LlamaParse</strong></td>
    <td style="text-align: center;">深度PDF结构解析</td>
    <td style="text-align: center;">法律合同、学术论文</td>
    <td style="text-align: center;">解析精度高，商业API</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Docling</strong></td>
    <td style="text-align: center;">模块化企业级解析</td>
    <td style="text-align: center;">企业合同、报告</td>
    <td style="text-align: center;">IBM生态兼容</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>Marker</strong></td>
    <td style="text-align: center;">PDF→Markdown，GPU加速</td>
    <td style="text-align: center;">科研文献、书籍</td>
    <td style="text-align: center;">专注PDF转换</td>
  </tr>
  <tr>
    <td style="text-align: center;"><strong>MinerU</strong></td>
    <td style="text-align: center;">多模态集成解析</td>
    <td style="text-align: center;">学术文献、财务报表</td>
    <td style="text-align: center;">集成LayoutLMv3+YOLOv8</td>
  </tr>
</table>
</div>

### 2.文本分块

#### 2.1 什么是文本分块？

	文本分块就是将加载后的长篇文档，切分成更小、更易于处理的单元，这些被切分出的文本块，是后续向量检索和模型处理的**基本单位**。

#### 2.2 为什么需要分块？

将文本分块的首要原因，是为了适应 RAG 系统中两个核心组件的硬性限制：

- **嵌入模型 (Embedding Model)**: 负责将文本块转换为向量。这类模型有严格的输入长度上限。例如，许多常用的嵌入模型（如 `bge-base-zh-v1.5`）的上下文窗口为512个token。任何超出此限制的文本块在输入时都会被截断，导致信息丢失，生成的向量也无法完整代表原文的语义。因此，文本块的大小**必须**小于等于嵌入模型的上下文窗口。
- **大语言模型 (LLM)**: 负责根据检索到的上下文生成答案。LLM同样有上下文窗口限制，检索到的所有文本块，连同用户问题和提示词，都必须能被放入这个窗口中，如果单个块过大，可能会导致只能容纳少数几个相关的块，限制了LLM回答问题时可参考的信息广度。

因此，分块是确保文本能够被两个模型完整、有效处理的基础。

#### 2.3 分块是不是越大越好？

**块的大小并非越大越好**，过大的块会严重影响RAG系统的性能。

##### （1）嵌入过程中的信息损失

大多数嵌入模型都基于 Transformer 编码器。其工作流程大致如下：

- **分词 (Tokenization)**: 将输入的文本块分解成一个个 token。
- **向量化 (Vectorization)**: Transformer 为**每个 token** 生成一个高维向量表示。
- **池化 (Pooling)**: 通过某种方法，如取 `[CLS]` 位的向量、对所有token向量求平均 `mean pooling` 等，将所有 token 的向量**压缩**成一个**单一的向量**，这个向量代表了整个文本块的语义。

> `[CLS]` 是BERT等Transformer模型在输入文本开头添加的特殊标记，通过自注意力机制动态聚合整个序列的上下文信息，其最终向量被训练用作代表全局语义的嵌入。

在这个`压缩`过程中，信息损失是不可避免的，**文本块越长，包含的语义点越多，这个单一向量所承载的信息就越稀释**，导致其表示变得笼统，关键细节被模糊化，从而降低了检索的精度。

##### （2）生成过程的“大海捞针” 

	即使将检索到的多个大块文本都塞进LLM的长上下文窗口中，也会出现关键信息被“淹没”在大量无关内容里的问题。有研究表明 ，当LLM处理非常长的、充满大量信息的上下文时，它倾向于更好地记住开头和结尾的信息，而忽略中间部分的内容。如果提供给LLM的上下文块又大又杂，模型就很难从中提取出最关键的信息来形成答案，从而导致回答质量下降或产生幻觉。

##### （3）主题稀释导致检索失败

	一个好的文本块应该聚焦于一个明确、单一的主题，如果一个块包含太多不相关的主题，它的语义就会被稀释，导致在检索时无法被精确匹配。

#### 2.4 常见分块方法

##### (1).Unstructured：基于文档元素的智能分块

（1）**分区 (Partitioning)**: 负责将原始文档（如PDF、HTML）解析成一系列结构化的“元素”（Elements）。每个元素都带有语义标签，如 `Title` (标题)、`NarrativeText` (叙述文本)、`ListItem` (列表项) 等。

（2）**分块 (Chunking)**: 该功能建立在**分区**的结果之上。分块功能不是对纯文本进行操作，而是将分区产生的“元素”列表作为输入，进行智能组合。Unstructured 提供了两种主要的分块方法：

- **`basic`**: 这是默认方法。这种方法会连续地组合文档元素（如段落、列表项），直到达到 `max_characters` 上限，尽可能地填满每个块。如果单个元素超过上限，则会对其进行文本分割。
- **`by_title`**: 该方法在 `basic` 方法的基础上，增加了对“章节”的感知。该方法将 `Title` 元素视为一个新章节的开始，并强制在此处开始一个新的块，确保同一个块内不会包含来自不同章节的内容。这在处理报告、书籍等结构化文档时非常有用，效果类似于 LangChain 的 `MarkdownHeaderTextSplitter`，但适用范围更广。

Unstructured 允许将分块作为分区的一个参数在单次调用中完成，也支持在分区之后作为一个独立的步骤来执行分块。这种“先理解、后分割”的策略，使得 Unstructured 能在最大程度上保留文档的原始语义结构，特别是在处理版式复杂的文档时，优势尤为明显。

##### (2).LlamaIndex：面向节点的解析与转换

[LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/) 将数据处理流程抽象为对“**节点（Node）**”的操作。文档被加载后，首先会被解析成一系列的“节点”，分块只是节点转换中的一环。LlamaIndex 的分块体系有以下特点：

（1）**丰富的节点解析器 (Node Parser)**: LlamaIndex 提供了大量针对特定数据格式和方法的节点解析器，可以大致分为几类：

- **结构感知型**: 如 `MarkdownNodeParser`, `JSONNodeParser`, `CodeSplitter` 等，能理解并根据源文件的结构（如Markdown标题、代码函数）进行切分。
- **语义感知型**:
  - `SemanticSplitterNodeParser`: 与 LangChain 的 `SemanticChunker` 类似，这种解析器使用嵌入模型来检测句子之间的语义“断点”，在语义连续性明显减弱的地方切开，从而让每个 chunk 内部尽量连贯。
  - `SentenceWindowNodeParser`: 这是一种巧妙的方法。该方法将文档切分成单个的句子，但在每个句子节点（Node）的元数据中，会存储其前后相邻的N个句子（即“窗口”）。这使得在检索时，可以先用单个句子的嵌入进行精确匹配，然后将包含上下文“窗口”的完整文本送给LLM，极大地提升了上下文的质量。
- **常规型**: 如 `TokenTextSplitter`, `SentenceSplitter` 等，提供基于Token数量或句子边界的常规切分方法。

（2）**灵活的转换流水线**: 用户可以构建一个灵活的流水线，例如先用 `MarkdownNodeParser` 按章节切分文档，再对每个章节节点应用 `SentenceSplitter` 进行更细粒度的句子级切分。每个节点都携带丰富的元数据，记录着其来源和上下文关系。

（3）**良好的互操作性**: LlamaIndex 提供了 `LangchainNodeParser`，可以方便地将任何 LangChain 的 `TextSplitter` 封装成 LlamaIndex 的节点解析器，无缝集成到其处理流程中。

### 三、向量嵌入与索引优化

#### 3.1 什么是向量嵌入

#### 3.2 怎么训练嵌入模型

#### 3.3 怎么选择合适的嵌入模型

#### 3.4 多模态嵌入

#### 3.5 常见向量数据库

#### 3.6 索引优化

# 🧠 Finetuning sentence transformer all-mpnet-base-v2 on Ray Technical Docs for RAG Pipelines

Embedding models are the backbone of modern Retrieval Augmented Generation pipelines, supplying a language model with the most similar and relevant context from a knowledgebase to aide it's generation.

More often than not, we default to standard and generalized embedding models to convert our data into dense vector representations, which are then stored in a vector database and retrieved at runtime. And while these models are quite powerful to start, they suffer in performance when applied to domain specific or niche content- often failing to retrieve the most relevant or useful documents from an end user perspective. This error compounds as it is passed to a language model, which will confidently answer with erroneous data.

This project fine-tunes the [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model from the [Sentence-Transformers](https://www.sbert.net/) library on domain-specific technical documentation (Ray.io docs). The goal is to enhance semantic retrieval quality in Retrieval-Augmented Generation (RAG) systems.

---

## 📌 Objectives

- Fine-tune an embedding model for domain-specific similarity retrieval.
- Deploy trained model and dataset to Hugging Face for reuse.
- Enable integration into RAG pipelines with improved relevance and precision.

---

## 🗃️ Dataset

I programmatically crawled and processed technical documentation from Ray.io using Python. My pipeline included:

  -Link extraction: fetched URLs from Ray’s sidebar navigation.
  
  -Selective HTML parsing: loaded only the main article content (<article class="bd-article">) to avoid unrelated page elements.
  
  -Content cleaning: removed emojis, excessive whitespace, repeated symbols, and unwanted characters to produce clean text data.
  
  -Document chunking: split long documents into overlapping chunks (default 1000 characters with 200-character overlap) using RecursiveCharacterTextSplitter.
  
  -Parallel loading: leveraged multithreading to accelerate page fetching and parsing.
  
  -File saving: saved cleaned documents as individual .txt files, ensuring safe filenames and avoiding collisions.
  
  -Filtering: discarded short chunks with fewer than 50 words to improve downstream model quality.

For each chunk, generated exactly 4 factual QA pairs using GPT-4.1:

  -Each question is answerable solely from the chunk’s content (no external knowledge).
  
  -Focuses on concrete facts rather than subjective interpretations.
  
  -Answers quote text verbatim from the chunk.
  
  -Post-processing and filtering:
  
  -Calculated cosine similarity between question embeddings and chunk embeddings using the OpenAI text-embedding-3-large model.
  
  -Removed any QA pairs where question-chunk similarity fell below a configurable threshold (default = 0.5).
  
  -Detected and removed duplicate or near-duplicate questions within the same chunk based on pairwise cosine similarity (default threshold = 0.78).

Use Cases:

  -Training domain-specific embedding models.
  
  -Fine-tuning retrieval or reranking systems for technical QA.
  
  -Benchmarking RAG performance on Ray technical content.

Columns:

  global_chunk_id: unique ID for each chunk.
  
  text: chunk content extracted from Ray documentation.
  
  question_id: index of each question (1-4).
  
  question: generated question.
  
  answer: exact text span from the chunk that answers the question.
  
This dataset provides a clean, high-quality resource for building domain-aware RAG pipelines on Ray’s technical domain.


📁 Available on Hugging Face Datasets:

➡️ [`thanhpham1/ray-technical-docs-qa-chunks`](https://huggingface.co/datasets/thanhpham1/ray-technical-docs-qa-chunks)

---

## 🔍 Model

We fine-tuned the all-mpnet-base-v2 sentence embedding model on synthetic question–chunk pairs extracted from Ray documentation.

Our fine-tuning used Matryoshka Representation Learning (MRL), combining MultipleNegativesRankingLoss with a MatryoshkaLoss wrapper to enable dynamic embedding truncation for adaptive retrieval.

We evaluated the model on various embedding dimensions (768, 512, 256, 128, 64) using Information Retrieval metrics (NDCG, MRR, MAP, Precision, Recall).

The fine-tuned model was published to Hugging Face Hub for downstream RAG tasks.

📦 Available on Hugging Face Models:

➡️ [`thanhpham1/Finetuned-all-mpnet-base-v2-with-technical-docs`](https://huggingface.co/thanhpham1/Finetuned-all-mpnet-base-v2-with-technical-docs)

---

## 📊 Evaluation

We evaluated the fine-tuned model on a synthetic question-answer dataset derived from Ray documentation.  
Evaluation was performed using **Information Retrieval** metrics across multiple embedding dimensions (768, 512, 256, 128, 64), leveraging **Matryoshka Representation Learning**.

### Metrics:

- **NDCG@10**
- **MRR@10**
- **MAP@100**
- **Accuracy@k**, **Precision@k**, **Recall@k** 
Example evaluation results:

| Metric   | Dimension | Base    | Fine-tuned | Abs. Improvement | % Improvement |
|----------|-----------|---------|------------|------------------|---------------|
| ndcg@10  | 768d      | 0.4926  | 0.7376     | 0.2450           | 49.7%         |
| ndcg@10  | 512d      | 0.4761  | 0.7283     | 0.2522           | 53.0%         |
| ndcg@10  | 256d      | 0.4222  | 0.7143     | 0.2921           | 69.2%         |
| ndcg@10  | 128d      | 0.3701  | 0.6903     | 0.3202           | 86.5%         |
| ndcg@10  | 64d       | 0.2671  | 0.5945     | 0.3274           | 122.6%        |
| mrr@10   | 768d      | 0.4232  | 0.6625     | 0.2393           | 56.5%         |
| mrr@10   | 512d      | 0.4063  | 0.6513     | 0.2450           | 60.3%         |
| mrr@10   | 256d      | 0.3640  | 0.6405     | 0.2765           | 76.0%         |
| mrr@10   | 128d      | 0.3111  | 0.6156     | 0.3045           | 97.9%         |
| mrr@10   | 64d       | 0.2199  | 0.5149     | 0.2950           | 134.2%        |
| map@100  | 768d      | 0.4682  | 0.6977     | 0.2295           | 49.0%         |
| map@100  | 512d      | 0.4521  | 0.6886     | 0.2365           | 52.3%         |
| map@100  | 256d      | 0.4071  | 0.6795     | 0.2724           | 66.9%         |
| map@100  | 128d      | 0.3547  | 0.6546     | 0.2999           | 84.6%         |
| map@100  | 64d       | 0.2604  | 0.5594     | 0.2990           | 114.8%        |

The fine-tuned model demonstrated significant improvements, especially at lower embedding dimensions — ideal for resource-constrained deployments.

---

## 🧪 How to Use

### Install dependencies:

```bash
%%capture
!pip install --upgrade sentence-transformers
!pip install git+https://github.com/huggingface/transformers
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub

model = SentenceTransformer("thanhpham1/Fine-tune-all-mpnet-base-v2", truncate_dim=256)
# Run inference
sentences = [
    'What type of framework is Ray described as?',
    '''Overview#
Ray is an open-source unified framework for scaling AI and Python applications like machine learning. It provides the compute layer for parallel processing so that you don’t need to be a distributed systems expert. Ray minimizes the complexity of running your distributed individual and end-to-end machine learning workflows with these components:

Scalable libraries for common machine learning tasks such as data preprocessing, distributed training, hyperparameter tuning, reinforcement learning, and model serving.
Pythonic distributed computing primitives for parallelizing and scaling Python applications.
Integrations and utilities for integrating and deploying a Ray cluster with existing tools and infrastructure such as Kubernetes, AWS, GCP, and Azure.

For data scientists and machine learning practitioners, Ray lets you scale jobs without needing infrastructure expertise:''', # Corresponding Positive
    '''Fault tolerance#Fault tolerance in Ray Train and Tune consists of experiment-level and trial-level
restoration. Experiment-level restoration refers to resuming all trials,
in the event that an experiment is interrupted in the middle of training due
to a cluster-level failure. Trial-level restoration refers to resuming
individual trials, in the event that a trial encounters a runtime
error such as OOM.

Framework#The deep-learning framework used for the model(s), loss(es), and optimizer(s)
inside an RLlib Algorithm. RLlib currently supports PyTorch and TensorFlow.

GCS / Global Control Service#Centralized metadata server for a Ray cluster. It runs on the Ray head node
and has functions like managing node membership and actor directory.
It’s also known as the Global Control Store.

Head node#A node that runs extra cluster-level processes like GCS and API server in
addition to those processes running on a worker node. A Ray cluster only has
one head node.''', # Random Excerpt
]

embeddings = model.encode(sentences)
print(embeddings.shape)

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities[0])
```

## ✅ Similarity Comparison

Below is an example comparing cosine similarities between a query and two passages from above, computed with the fine-tuned model vs. the original base model.


### 🔥 Fine-Tuned Model Similarities

| Pair                              | Cosine Similarity |
|-----------------------------------|-------------------|
| Query vs. Passage 1 (Relevant)    | **1.0000**        |
| Query vs. Passage 2 (Irrelevant)  | 0.5034            |
| Passage 1 vs. Passage 2           | 0.1761            |

The fine-tuned model produces a much higher similarity for the relevant pair and better separation from irrelevant content.

---

### 🧪 Base Model (Before Fine-Tuning) Similarities

| Pair                              | Cosine Similarity |
|-----------------------------------|-------------------|
| Query vs. Passage 1 (Relevant)    | **1.0000**        |
| Query vs. Passage 2 (Irrelevant)  | 0.3352            |
| Passage 1 vs. Passage 2           | 0.4369            |

The base model shows poorer separation, with a higher similarity to the irrelevant passage.

---

### 🚀 Summary

- The fine-tuned model **increased the relevant similarity margin** from `0.3352 → 0.5034`.
- It **lowered the unrelated pair similarity** from `0.4369 → 0.1761`.
- This demonstrates that fine-tuning helps the embedding model better distinguish relevant context in domain-specific retrieval tasks.


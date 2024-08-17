# Embedding Model

## Motivation
I wanted to recreate Word2Vec from scratch including:
- Sourcing data (English Wikipedia used in this)
- Cleaning data (To be further discussed)
- Creating a tokenizer (BPE algorithm based on Karpathy video, then further optimized for run-time)
- Training word embeddings (Continuous Bag of Words w/ Negative Sampling)
- Analyzing word embeddings (PCA, Cosine Similarity)

I also attempted to train a transformer using this tokenizer and these embeddings, but it didn't end up providing good results. I think it's because I couldn't use enough compute and we're dealing with a vocabulary of size 16k -- which leads to sparsity issues that would require large batch sizes that my machine can't handle.

## Included in this Repo
1. **Dataloader**: I created a dataloader that I can use across many of my NLP projects so you won't be able to use this in its current form. May make it a package but likely not.   
2. **Tokenizer**: Using Andrej Karpathy's tutorial as a starting point, I built an optimized bytepair encoding tokenizer (BPETokenizer). *I ended up using SentencePiece for efficiency purposes.*   
3. **Embedding Model**: Code for training and analyzing the embedding model. I used Continuous Bag of Words with Negative Sampling and played around a a bit with adding Hierarchal Softmax.   
4. **Transformer**: For fun, training a transformer using the trained embeddings and tokenizer. Didn't work out too well unfortunately...   


## Setup Instructions

### Creating the Environment
To set up the project environment, run the following command in the project directory (where `environment.yaml` is located):

```bash
conda env create -f environment.yaml
```
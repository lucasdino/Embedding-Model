# Embedding Model

## Motivation
The current research project I'm interested in involves trying to generate embeddings for sequences. While token-level embeddings have proven to be incredibly powerful, I think that thinking on a token-level basis presents limitations and you can build more powerful models by thinking on a sequence-level basis. This ultimately has interesting use cases for Neurosymbolic Programming, which is also why I am interested in pursuing this further.\n

## Included in this Repo
1. **Tokenizer**: Using Andrej Karpathy's tutorial as a starting point, I built an optimized bytepair encoding tokenizer (BPETokenizer).\n
- In this folder, you have **bpetokenizer** (my DIY version) and **sptokenizer** (the production version using SentencePiece).
- *Note: The DIY tokenizer was used as an exercise - we'll be using SentencePiece to train a bespoke tokenizer on English text from Wikipedia. This is because of compute limitations; too large of a token vocabulary means a larger embedding model when training (and will require more data to see enough examples to learn meaningful representations for each token). Rather than a vocabulary size of ~50k, I'll likely cap it at ~5k.*
2. **Embedding Model**: Training an embedding model using the tokenizer trained in *sptokenizer*. We'll implement a few different approaches (CBOW, Skipgram) to test performance on a test that I'll define.
3. **Transformer**: For fun, training a transformer using our bespoke tokenizer and embedding model to see how well we can do. We may build upon this further to see if I can 


## Setup Instructions

### Creating the Environment
To set up the project environment, run the following command in the project directory (where `environment.yml` is located):

```bash
conda env create -f environment.yml
```
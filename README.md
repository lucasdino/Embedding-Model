# Embedding Model
---
In the process of learning about embeddings, I'm building my own embedding model based on Word2Vec. Included:
1. **Tokenizer**: Using Andrej Karpathy's tutorial as a starting point, I built an optimized bytepair encoding tokenizer (BPETokenizer). *This was more as an exercise - we'll be using the GPT2Tokenizer from Hugging Face's Transformer library to encode our UTF-8 text.*
2. **Embedding Model**: Training an embedding model based on Word2Vec and other common methods. Using defined tokenizer for my implementations.
3. **GPT**: Training a transformer to see how effective I can model tinyshakespeare and wikipedia text.
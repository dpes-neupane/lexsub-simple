# Simple Lexical Substitution Algorithm Using Word Embeddings

The `lexsub.py` script consist of a very crude way to get some substitutions for a given __target_word__  

The **algorithm** can be summarized as:

> 1. Get the word embeddings.
> 2. Find the nearest words in the vector space of the given target word using the euclidean distance (also, using cosine distance is also a valid idea)
> 3. Get the target sentence's embeddings as the average of its constituent words' embeddings.
> 4. Similarly, replace the target word from the target sentence and compute the sentence embeddings with candidate words that we got from **Step 2**.
> 5. Compute the dot product for each sentence embeddings with candidate words and target word.
> 6. Get **k** largest valued dot product substitute words from the given candidate words.

Example to run the script:
```
py lexsub.py -t "mission" -s "A mission to end a war AUSTIN, Texas -- Tom Karnes was dialing for destiny but not everyone wanted to cooperate" -n 100 -o 10 -p "dot"

```


For more better lexical substitution works, please read [Nikolay Arefyev et al.](https://arxiv.org/pdf/2006.00031.pdf), [Roller and Katrin](https://aclanthology.org/N16-1131.pdf), [Michalopoulos et al.](https://aclanthology.org/2022.acl-long.87.pdf) and [Zhou1 et al.](https://aclanthology.org/P19-1328.pdf)

The `emb30010k/` contains two `.tsv` files:
- `metadata.tsv` which contains approx most common 10k words
- `vectors.tsv` contains the vectors of those 10k words trained using SGNS (pretrained gensim vectors downloaded from [SGNS](http://vectors.nlpl.eu/repository/))  
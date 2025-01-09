catGPT more like CARGPT
RAG model made from scratch using llama3.2 1B as LANGUAGE MODEL (hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF) and COMPENDIUM LABS EMBEDDING MODEL (hf.co/CompendiumLabs/bge-base-en-v1.5-gguf)
Made custom VECTOR_DB to get the similarites
Implemented cosine similarities from scratch


.. takes user query --> 
.. optimizes the query with language model --> 
.. optimized query is then embedded and those embeddings are matched with vector DB to get the relevant chunks for language model -->
.. with relevant chunks and query, it's fed in Language model with a designed prompt template to extract the imformation.

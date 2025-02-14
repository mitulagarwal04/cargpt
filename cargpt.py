import ollama
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)


def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

## creating cosine similarity function
def consine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a,b)])
    norm_a = sum([x**2 for x in a]) ** 0.5
    norm_b = sum([x**2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

## make the model generate the query itself so that it picks up diverse information related to the user query
def refine_query(user_query):
    instruction = f"""
    You are an assistant helping to refine questions. Rewrite the user's input query into smaller, relevant sub-queries :
    User query: {user_query}
    Refined queries:
    """
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": "Refine the query. Keep it simple and return only the queries."},
            {"role": "user", "content": instruction},
        ],
    )   
    # return (response)
    refined_queries = response['message']['content'].strip().split('\n')
    return [query.strip('- ') for query in refined_queries if query.strip()]

## retrieve similar chunks 
def retrieve(user_query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=user_query)['embeddings'][0]
    
    similarities = []
    for chunk, embedding in (VECTOR_DB):
        similarity = consine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x:x[1], reverse=True)
    return similarities[:top_n]

## aggregrate all retireved chunks and get query
def aggregrated_retrieval(user_query, top_n=3):
    refined_queries = refine_query(user_query)
    all_results = []
    for sub_query in refined_queries:
        results = retrieve(sub_query, top_n=top_n)
        all_results.extend(results)

    seen_chunks = set()
    unique_results = []
    for chunk, similarity in all_results:
        if chunk not in seen_chunks:
            unique_results.append((chunk, similarity))
            seen_chunks.add(chunk)

    unique_results.sort(key = lambda x: x[1], reverse=True)
    return unique_results[:top_n] 

def rerank_results(query, retrieved_chunks):
    pairs = [(query, chunk) for chunk,_ in retrieved_chunks]
    inputs = tokenizer(pairs, truncation=True, padding=True, return_tensors='pt')

    with torch.no_grad():
        scores = reranker(**inputs).logits.squeeze(1).tolist()

    rerank_results = sorted(
        zip([chunk for chunk, _ in retrieved_chunks], scores),
        key=lambda x:x[1],
        reverse=True
    )
    return ranked_results

def retrieve_with_rerank(user_query, top_n=3):
    query_embedding = ollama.embed(model = EMBEDDING_MODEL, input=user_query)['embeddings'][0]

    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = consine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    initial_top_chunks = similarities[:top_n]

    reranked_chunks = rerank_results(user_query, initial_top_chunks)
    
    return reranked_chunks


## creating VECTOR_DB, embedding input text
dataset = []
with open('./dataset/cat_facts.txt', 'r', encoding="utf8") as file:
    dataset = file.readlines()

VECTOR_DB = []

for idx, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    

## generation phase
input_query = input("Ask a question about cats: ")
# input_query = 'What are cats'

retrieved_knowledge = aggregrated_retrieval(input_query, 3)

print('Retrieved knowledge:')
print(retrieved_knowledge)
print('input queries: ')
print(refine_query(input_query))

instruction_prompt = f"""You are a helpful chatbot.
Use only the following context to answer the question. Ensure your answer is directly based on the provided context. 
You may rephrase or organize the information if needed, but do not introduce any external knowledge. 
If the answer cannot be reasonably inferred from the provided context, respond with "I don't know."
Context:
{chr(20).join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
"""


print(instruction_prompt)

stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages = [
        {'role':'system', 'content':instruction_prompt},
        {'role':'user', 'content':input_query}
    ],
    stream = True
)

print(f"chatbot response: ")

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)







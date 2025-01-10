from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"

tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)

query = "What are the benefits of renewable energy?"
doc = "Renewable energy sources like solar and wind can reduce pollution."

inputs = tokenizer(query, doc, return_tensors="pt", padding=True, truncation=True)

outputs = reranker(**inputs)
scores = outputs.logits

print(outputs)
print(scores)
import psycopg2
import os
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "intfloat/multilingual-e5-large"

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto")

con = psycopg2.connect(host="localhost",
                       user=os.getenv("POSTGRES_USER"),
                       password=os.getenv("POSTGRES_PASSWORD"),
                       database=os.getenv("POSTGRES_DB"))

cur = con.cursor()

while True:
  q = input("Input query: ")
  input_ids = tokenizer("query: " + q, max_length=512, padding=True, truncation=True, return_tensors='pt')
  outputs = model(**input_ids)
  embeddings = average_pool(outputs.last_hidden_state, input_ids['attention_mask'])
  sql_select = f"""SELECT
    text_content, 
    1 - (embedding <=> '{str(embeddings[0].tolist())}') AS cosine_similarity
    FROM embedding_store
    ORDER BY 2 DESC"""
  cur.execute(sql_select)
  for r in cur.fetchall():
    print(r[1], r[0][0:100])

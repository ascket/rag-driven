# %%
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
import deeplake.util
import os
from dotenv import load_dotenv

load_dotenv()

ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")

if "ACTIVELOOP_TOKEN" not in os.environ:
    os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN

# Путь к векторному хранилищу. Если его нет, то оно будет создано
vector_store_path = "hub://so434/space_exploration_v1"

try:
    # Попытка загрузки векторной базы данных
    vector_store = VectorStore(path=vector_store_path)
    print("Vector store exists")
except FileNotFoundError:
    print("Vector store does not exist. You can create it.")
    # Код для создания векторной базы данных
    create_vector_store = True

# %%
source_text = "retriever/llm.txt"
with open(source_text, 'r', encoding='utf-8') as f:
    main_text = f.readlines()
    CHUNK_SIZE = 1000
    text = main_text[0]
    chunked_text = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    print(chunked_text)

# %%
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPEN_AI_PAY_KEY")

# if "OPENAI_API_KEY" not in os.environ:
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-ada-002', )


def embedding_function(texts):
    texts = [text.replace('\n', ' ') for text in texts]
    embed = embeddings.embed_documents(texts=texts)
    return embed


# with open(source_text, 'r', encoding='utf-8') as f:
#     main_text = f.readlines()
#     text = main_text[0]
#     print(embedding_function(text))

# %%

add_to_vector_store = True
if add_to_vector_store:
    with open(source_text, 'r', encoding='utf-8') as f:
        text = f.read()
        CHUNK_SIZE = 1000
        chunked_text = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

vector_store.add(
    text=chunked_text,
    embedding_function=embedding_function,
    embedding_data=chunked_text,
    metadata=[{"source": source_text}] * len(chunked_text)
)

#%%
deeplake.delete(vector_store_path)
#Query: https://docs.deeplake.ai/4.4/api/dataset


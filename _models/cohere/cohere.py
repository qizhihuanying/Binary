import os
import math
import cohere
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

# 将客户端初始化延迟到实际使用时
co = None

def get_cohere_client():
    global co
    if co is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        co = cohere.Client(api_key)
    return co

def get_cohere_embedding(
    prompt: str, model: str = "embed-english-light-v3.0", input_type: str = "search_document"
) -> list[float]:
    client = get_cohere_client()
    response = client.embed(texts=[prompt], model=model, input_type=input_type)
    return response.embeddings[0]

def get_cohere_embeddings_batched(
    prompts: list[str],
    batch_size: int = 96,
    model: str = "embed-english-light-v3.0",
    input_type: str = "search_document",
    pbar: bool = False,
) -> list[list[float]]:
    assert batch_size <= 96, "cohere limits batch size at 96"
    
    client = get_cohere_client()
    num_batches = math.ceil(len(prompts) / batch_size)
    embeddings = []
    for i in tqdm(range(num_batches), disable=not pbar):
        inputs = prompts[i * batch_size : (i + 1) * batch_size]
        response = client.embed(texts=inputs, model=model, input_type=input_type, truncate="END")
        embedding = response.embeddings
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)
    embeddings = embeddings.tolist()
    return embeddings
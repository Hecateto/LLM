from langchain_community.retrievers import BM25Retriever
from typing import List
import jieba
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

loader = TextLoader('medical_data.txt', encoding='utf-8')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n"]
)
docs = text_splitter.split_documents(documents)

def preprocessing_func(text: str) -> List[str]:
    return list(jieba.cut(text))    # 使用jieba分词进行中文分词

'''
稀疏检索
依赖词汇重叠计算文档和查询之间的相关性
将文本表示为高维稀疏向量,无法理解语义相似性
'''
# 稀疏检索 - BM25
bm25 = BM25Retriever(docs=docs, k=10)
print(f"BM25.k: {bm25.k}")
retriever = bm25.from_documents(docs, preprocess_func=preprocessing_func)
print(retriever.invoke("什么水果可以清血脂?"))
# List[Document], Document: {metadata, page_content: {{question, answer}, ...}, }

# 稀疏检索 - BM25Okapi
from rank_bm25 import BM25Okapi
texts = [i.page_content for i in docs]
texts_processed = [preprocessing_func(text) for text in texts]
vectorizer = BM25Okapi(texts_processed)
bm25_res = vectorizer.get_top_n(preprocessing_func("什么水果可以清血脂?"), texts, n=3)
print(bm25_res)
# [{question, answer}, ...]

'''
密集检索
使用DL模型将查询和文档映射到低维密集的连续向量空间
通过计算向量的距离或相似度（e.g. 余弦相似度）衡量相关性
'''
# FAISS
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-large-zh-v1.5',
    model_kwargs={'device': 'cuda:0'}
)
db = FAISS.from_documents(docs, embeddings)
vector_res = db.similarity_search("什么水果可以清血脂?", k=3)
print(vector_res)
# List[Document], Document: {id, metadata, page_content: {{question, answer}, ...}, }

def rrf(vector_results: List[str], text_results: List[str], k: int=3, m: int=60):
    '''
    Reciprocal Rank Fusion
    融合稀疏检索和密集检索的结果
    '''
    docs_scores = {}
    for rank, doc_id in enumerate(vector_results):
        docs_scores[doc_id] = docs_scores.get(doc_id, 0) + 1 / (rank + m)
    for rank, doc_id in enumerate(text_results):
        docs_scores[doc_id] = docs_scores.get(doc_id, 0) + 1 / (rank + m)

    sorted_res = [d for d, _ in sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)[:k]]
    return sorted_res

vector_results = [i.page_content for i in vector_res]
text_results = [i for i in bm25_res]
rrf_res = rrf(vector_results, text_results, k=3, m=60)
print(rrf_res)

# LLM
prompt = '''
任务目标：根据检索出的文档回答用户问题任务要求：    
1、不得脱离检索出的文档回答问题    
2、若检索出的文档不包含用户问题的答案，请回答我不知道用户问题：{}检索出的文档：{}
'''

base_url = ''
api_key = ''
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model='Qwen/Qwen2.5-7B-Instruct',
    base_url=base_url,
    api_key=api_key) # 初始化聊天模型
# 将用户问题和RRF融合后的检索结果填充到prompt模板中
res = model.invoke(prompt.format('什么水果可以清血脂?', ''.join(rrf_res)))
print(res.content)
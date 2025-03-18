import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Callable, Union, Dict, Tuple
from functools import partial
import numpy as np
from sklearn.metrics import ndcg_score
from collections import defaultdict
import pandas as pd

from binary import BinaryHead

class MultiModelBinaryTrainer:
    def __init__(
        self,
        models: List[nn.Module],
        tokenizers: List[any],
        embedding_funcs: List[Callable],
        device: torch.device,
        temp: float = 1.0,
        num_trainable_layers: int = 2,
        lr: float = 2e-5,
        use_binary_head: bool = True
    ):
        self.models = models  # 用于直接获取本地embedding的模型
        self.tokenizers = tokenizers
        self.embedding_funcs = embedding_funcs  # 用于API方式获取embedding的函数
        self.device = device
        self.binary_head = BinaryHead(temp=temp, use_binary_head=use_binary_head).to(device)
        self.criterion = nn.CosineEmbeddingLoss()
        
        for model in self.models:
            self._prepare_model_for_training(model, num_trainable_layers)
            
        params = list(self.binary_head.parameters())
        if num_trainable_layers > 0:
            for model in self.models:
                params.extend(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = optim.Adam(params, lr=lr)

    def _prepare_model_for_training(self, model, num_trainable_layers):
        """准备模型训练，冻结或解冻相应层"""
        for param in model.parameters():
            param.requires_grad = False
        
        if num_trainable_layers > 0:
            if hasattr(model, 'encoder'):
                encoder_layers = model.encoder.layer
            elif hasattr(model, 'layers'):
                encoder_layers = model.layers
            else:
                raise ValueError("Model architecture not supported for partial training")
            
            for layer in encoder_layers[-num_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_embeddings(self, texts, model_idx):
        """统一获取embedding的接口"""
        if model_idx < len(self.models):  # 使用本地模型
            model = self.models[model_idx]
            tokenizer = self.tokenizers[model_idx]
            
            enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model(**enc)
                emb = torch.nn.functional.normalize(out[0][:, 0], p=2, dim=1)
            # 确保返回的张量有梯度
            emb = emb.detach().clone().requires_grad_(True)
            return emb
        else:  # 使用API方式获取embedding
            func_idx = model_idx - len(self.models)
            embeddings = self.embedding_funcs[func_idx](prompts=texts)
            # 确保返回的张量有梯度
            emb = torch.tensor(embeddings, device=self.device, requires_grad=True)
            return emb

    def train_epoch(self, data, batch_size, current_iter, total_iters):
        for model in self.models:
            if any(p.requires_grad for p in model.parameters()):
                model.train()
            else:
                model.eval()
        self.binary_head.train()
        
        total_loss = 0.0
        num_batches = max(int(len(data) / batch_size + 0.99), 1)
        data_shuffled = data.sample(frac=1, random_state=42)

        for i in tqdm(range(num_batches), desc="Training"):
            current_temp = max(0.01, 1.0 - (current_iter + i) / total_iters)
            self.binary_head.temp = current_temp
            batch = data_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_loss = 0
            
            # 对每个模型/函数计算embeddings并累积loss
            for idx in range(len(self.models) + len(self.embedding_funcs)):
                emb1 = self.get_embeddings(list(batch["sample1"]), idx)
                emb2 = self.get_embeddings(list(batch["sample2"]), idx)
                
                # 从relevance动态计算pos值：相关性>0为正例(1)，否则为负例(-1)
                labels = torch.tensor([1 if rel > 0 else -1 for rel in batch["relevance"].values]).float().to(self.device)
                
                if not self.binary_head.use_binary_head:
                    batch_loss += self.criterion(emb1, emb2, labels)
                else:
                    bin_out1 = self.binary_head(emb1)
                    bin_out2 = self.binary_head(emb2)
                    batch_loss += self.criterion(bin_out1, bin_out2, labels)

            # 平均所有模型的loss
            loss = batch_loss / (len(self.models) + len(self.embedding_funcs))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(data)

    def eval_epoch(self, data, batch_size):
        self.binary_head.eval()
        for model in self.models:
            model.eval()
            
        total_loss = 0.0
        num_batches = max(int(len(data) / batch_size + 0.99), 1)

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Evaluating"):
                batch = data[i * batch_size : (i + 1) * batch_size]
                batch_loss = 0
                
                for idx in range(len(self.models) + len(self.embedding_funcs)):
                    emb1 = self.get_embeddings(list(batch["sample1"]), idx)
                    emb2 = self.get_embeddings(list(batch["sample2"]), idx)
                    
                    # 从relevance动态计算pos值
                    labels = torch.tensor([1 if rel > 0 else -1 for rel in batch["relevance"].values]).float().to(self.device)

                    out1 = self.binary_head(emb1)
                    out2 = self.binary_head(emb2)
                    batch_loss += self.criterion(out1, out2, labels)

                total_loss += batch_loss.item() / (len(self.models) + len(self.embedding_funcs))

        return total_loss / len(data)
        
    def calculate_ndcg10(self, data: pd.DataFrame) -> dict:
        
        self.binary_head.eval()
        for model in self.models:
            model.eval()

        results = {}

        # 如果有 lang 列，就按语言区分，否则统一当做 'all'
        if 'lang' in data.columns:
            languages = data['lang'].unique()
        else:
            languages = ['all']
            data['lang'] = 'all'

        for lang in languages:
            lang_data = data[data['lang'] == lang]
            results[lang] = {}

            # 模型数量（本地模型 + embedding_funcs）
            total_model_count = len(self.models) + len(self.embedding_funcs)

            for model_idx in range(total_model_count):
                model_name = f"model_{model_idx}"
                print(f"\n开始计算 {lang} 语言模型 {model_idx} 的NDCG@10")
                
                # 收集所有唯一的查询和文档
                queries = {}         # 查询ID -> 查询文本
                documents = {}       # 文档ID -> 文档文本
                qrels = {}           # 查询ID -> {文档ID: 相关性}
                
                # 1. 首先收集所有唯一的查询和标注的文档关系
                for _, row in tqdm(lang_data.iterrows(), desc="收集查询和标注数据", total=len(lang_data)):
                    query_id = row['query_id']
                    doc_id = row['doc_id']
                    query_text = row['sample1']
                    doc_text = row['sample2']
                    relevance = row['relevance']
                    
                    queries[query_id] = query_text
                    documents[doc_id] = doc_text
                    
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = relevance
                
                # 2. 收集语料库中所有文档（这里我们使用已有数据集中的所有文档作为语料库）
                # 在实际应用中，你可能需要加载完整的语料库文件
                corpus_docs = documents  # 使用已有的所有文档作为语料库
                
                print(f"收集到 {len(queries)} 个唯一查询和 {len(corpus_docs)} 个语料库文档")
                
                # 批量计算所有查询的embedding
                query_embeddings = {}  # 查询ID -> embedding
                batch_size = 32  # 可以根据GPU内存调整
                
                # 将查询ID和文本转换为列表，以便批处理
                query_ids = list(queries.keys())
                query_texts = [queries[qid] for qid in query_ids]
                
                # 批量计算查询embedding
                for i in tqdm(range(0, len(query_texts), batch_size), desc="计算查询embedding"):
                    batch_texts = query_texts[i:i+batch_size]
                    batch_ids = query_ids[i:i+batch_size]
                    
                    with torch.no_grad():
                        batch_embs = self.get_embeddings(batch_texts, model_idx)
                        if self.binary_head.use_binary_head:
                            batch_embs = self.binary_head(batch_embs)
                    
                    # 将embedding保存到字典中
                    for j, qid in enumerate(batch_ids):
                        query_embeddings[qid] = batch_embs[j:j+1]
                
                # 批量计算所有语料库文档的embedding
                corpus_embeddings = {}  # 文档ID -> embedding
                
                # 将文档ID和文本转换为列表，以便批处理
                corpus_doc_ids = list(corpus_docs.keys())
                corpus_doc_texts = [corpus_docs[did] for did in corpus_doc_ids]
                
                # 批量计算文档embedding
                for i in tqdm(range(0, len(corpus_doc_texts), batch_size), desc="计算语料库文档embedding"):
                    batch_texts = corpus_doc_texts[i:i+batch_size]
                    batch_ids = corpus_doc_ids[i:i+batch_size]
                    
                    with torch.no_grad():
                        batch_embs = self.get_embeddings(batch_texts, model_idx)
                        if self.binary_head.use_binary_head:
                            batch_embs = self.binary_head(batch_embs)
                    
                    # 将embedding保存到字典中
                    for j, did in enumerate(batch_ids):
                        corpus_embeddings[did] = batch_embs[j:j+1]
                
                # 计算每个查询的NDCG@10
                ndcg_list = []
                
                for query_id in tqdm(queries.keys(), desc="计算NDCG@10"):
                    if query_id not in query_embeddings:
                        print(f"警告: 查询 {query_id} 没有embedding")
                        continue
                    
                    query_emb = query_embeddings[query_id]
                    
                    # 计算查询与语料库中每个文档的相似度
                    doc_similarities = []  # [(doc_id, similarity), ...]
                    
                    for doc_id, doc_emb in corpus_embeddings.items():
                        sim = torch.nn.functional.cosine_similarity(query_emb, doc_emb).item()
                        doc_similarities.append((doc_id, sim))
                    
                    # 按相似度降序排序
                    doc_similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    # 取前K个进行评估
                    k = 10
                    
                    # 构建真实相关性列表（对所有文档，未标注的文档相关性为0）
                    y_true = []
                    y_score = []
                    for doc_id, sim in doc_similarities:
                        relevance = qrels.get(query_id, {}).get(doc_id, 0)
                        y_true.append(relevance)
                        y_score.append(sim)
                    
                    # 计算NDCG@10
                    try:
                        ndcg_val = ndcg_score(
                            y_true=[y_true],  # 真实相关性
                            y_score=[y_score],  # 预测相似度
                            k=k
                        )
                        ndcg_list.append(ndcg_val)
                        
                        # 打印一些样本的详细信息以便调试
                        if len(ndcg_list) <= 3:  # 只打印前3个查询的详细信息
                            print(f"\n查询ID: {query_id}")
                            print(f"查询文本: {queries[query_id]}")
                            print(f"Top {k} 结果:")
                            for i, (doc_id, sim) in enumerate(doc_similarities[:k]):
                                relevance = qrels.get(query_id, {}).get(doc_id, 0)
                                doc_snippet = corpus_docs[doc_id][:50] + "..." if len(corpus_docs[doc_id]) > 50 else corpus_docs[doc_id]
                                print(f"  {i+1}. 文档ID: {doc_id}, 相关性: {relevance}, 相似度: {sim:.4f}")
                                print(f"     文档片段: {doc_snippet}")
                            print(f"NDCG@{k}: {ndcg_val:.4f}")
                        
                    except Exception as e:
                        print(f"查询 {query_id} 计算NDCG出错: {e}")
                        continue
                
                # 计算该模型在当前语言上的平均NDCG@10
                if ndcg_list:
                    avg_ndcg = float(np.mean(ndcg_list))
                    results[lang][model_name] = avg_ndcg
                    print(f"\n{lang} 语言模型 {model_idx} 的平均NDCG@10: {avg_ndcg:.4f}")
                else:
                    results[lang][model_name] = 0.0
                    print(f"\n{lang} 语言模型 {model_idx} 没有有效的NDCG@10结果")

        return results

    def evaluate_retrieval(self, data: pd.DataFrame, batch_size: int) -> Dict[str, Dict[str, float]]:
        self.binary_head.eval()
        for model in self.models:
            model.eval()   
        with torch.no_grad():
            results = self.calculate_ndcg10(data)
        return results

    def train_binary_head(self, train_data, val_data, test_data, epochs, batch_size, output_dir):
        # 检查训练集是否为空
        if len(train_data) == 0:
            print("训练集为空，跳过训练过程，直接进行评估...")
            # 初始化binary_head为评估模式
            self.binary_head.training = False
            
            # 对测试集进行评估
            if test_data is not None and len(test_data) > 0:
                # 计算检索评估指标
                if 'query_id' in test_data.columns and 'relevance' in test_data.columns:
                    retrieval_results = self.evaluate_retrieval(test_data, batch_size)
                    print(f"检索评估结果 (NDCG@10):")
                    for lang, models_results in retrieval_results.items():
                        print(f"  语言: {lang}")
                        for model_name, ndcg in models_results.items():
                            print(f"    {model_name}: {ndcg:.4f}")
            return
        
        num_batches_per_epoch = max(int(len(train_data) / batch_size + 0.99), 1)
        total_iters = epochs * num_batches_per_epoch
        
        current_iter = 0
        best_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_data, batch_size, current_iter, total_iters)
            current_iter += num_batches_per_epoch
            
            # 检查验证集是否为空
            if val_data is not None and len(val_data) > 0:
                val_loss = self.eval_epoch(val_data, batch_size)
                print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Current Temp: {self.binary_head.temp:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    print(f"新的最佳验证损失: {best_loss:.4f}, 保存模型...")
                    self.binary_head.save_model(f"{output_dir}/best_model")
            else:
                print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Current Temp: {self.binary_head.temp:.4f}")
                # 没有验证集时，直接保存每个epoch的模型
                print(f"保存当前epoch模型...")
                self.binary_head.save_model(f"{output_dir}/epoch_{epoch+1}")

        print("重新加载最佳模型进行评估...")
        original_use_binary_head = self.binary_head.use_binary_head
        self.binary_head = BinaryHead.load_model(
            f"{output_dir}/binary_head_full.pt",
            self.device
        )
        self.binary_head.use_binary_head = original_use_binary_head
        self.binary_head.training = False
        print(f"评估设置：use_binary_head={self.binary_head.use_binary_head}, training={self.binary_head.training}")
        
        # 对测试集进行评估(损失和NDCG@10)
        # test_loss = self.eval_epoch(test_data, batch_size)
        # print(f"测试损失: {test_loss:.4f}")
        
        # 计算检索评估指标
        if test_data is not None and len(test_data) > 0 and 'query_id' in test_data.columns and 'relevance' in test_data.columns:
            retrieval_results = self.evaluate_retrieval(test_data, batch_size)
            print(f"检索评估结果 (NDCG@10):")
            for lang, models_results in retrieval_results.items():
                print(f"  语言: {lang}")
                for model_name, ndcg in models_results.items():
                    print(f"    {model_name}: {ndcg:.4f}")



def train(
    models: List[nn.Module],
    tokenizers: List[any],
    embedding_funcs: List[Callable],
    train_data,
    val_data=None,
    test_data=None,
    device: torch.device = None,
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 4,
    temp: float = 1.0,
    num_trainable_layers: int = 1,
    output_dir: str = "project/models/binary_head",
    use_binary_head: bool = True
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    trainer = MultiModelBinaryTrainer(
        models=models,
        tokenizers=tokenizers,
        embedding_funcs=embedding_funcs,
        device=device,
        temp=temp,
        num_trainable_layers=num_trainable_layers,
        lr=lr,
        use_binary_head=use_binary_head
    )
    
    trainer.train_binary_head(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )
        

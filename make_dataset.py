import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
import json
from typing import Dict, List, Tuple
import datasets
from datasets import load_dataset
import random

# MIRACL支持的语言列表
MIRACL_LANGUAGES = [
    "ar", "bn", "en", "es", "fi", "fr", "hi", "id", 
    "ja", "ko", "ru", "sw", "te", "th", "zh", "de", "yo", "fa"
]

def load_miracl_data(lang: str, split: str = "dev", force_hf: bool = False) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[Tuple[str, int]]]]:
    """
    加载MIRACL数据集，优先从本地加载，如果本地不存在或指定强制从HF加载，则从Hugging Face加载
    
    Args:
        lang: 语言代码
        split: 数据集拆分，'dev'或'train'
        force_hf: 是否强制从Hugging Face加载
        
    Returns:
        queries: 查询字典 {query_id: query_text}
        documents: 文档字典 {doc_id: doc_text}
        qrels: 相关性字典 {query_id: [(doc_id, relevance)]}
    """
    base_path = Path(f"datasets/miracl/{lang}/{split}")
    
    # 检查是否需要从Hugging Face加载
    if force_hf or not base_path.exists() or not all((base_path / file).exists() for file in ["corpus.jsonl", "queries.jsonl", "qrels.jsonl"]):
        print(f"从Hugging Face加载 {lang} 语言的MIRACL数据集 ({split})...")
        
        if lang not in MIRACL_LANGUAGES:
            raise ValueError(f"不支持的语言: {lang}，支持的语言有: {MIRACL_LANGUAGES}")
        
        if split not in ["dev", "train"]:
            raise ValueError(f"不支持的数据集拆分: {split}，支持的拆分有: ['dev', 'train']")
        
        try:
            # 加载查询和相关性数据
            dataset = load_dataset("miracl/miracl", f"{lang}", split=split)
            
            # 尝试加载文档集合
            try:
                corpus = load_dataset("miracl/miracl", f"{lang}", split="corpus")
            except:
                corpus = None
                print(f"无法加载 {lang} 语言的corpus分割，将尝试从{split}集提取文档")
            
            # 构建查询字典
            queries = {}
            for item in dataset:
                query_id = str(item["query_id"])
                query_text = item["query"]
                queries[query_id] = query_text
                
            # 构建文档字典
            documents = {}
            if corpus is not None:
                # 从corpus中加载文档
                for item in corpus:
                    doc_id = str(item["docid"])
                    doc_text = item["text"]
                    documents[doc_id] = doc_text
            else:
                # 从数据集中提取文档
                for item in dataset:
                    for passage_type in ["positive_passages", "negative_passages"]:
                        if passage_type in item:
                            for passage in item[passage_type]:
                                doc_id = str(passage["docid"])
                                doc_text = passage["text"]
                                documents[doc_id] = doc_text
                
            # 构建相关性判断字典
            qrels = {}
            for item in dataset:
                query_id = str(item["query_id"])
                
                # 添加正例（positive_passages）
                if "positive_passages" in item:
                    if query_id not in qrels:
                        qrels[query_id] = []
                    
                    for passage in item["positive_passages"]:
                        doc_id = str(passage["docid"])
                        qrels[query_id].append((doc_id, 1))  # 相关性为1
                
                # 添加负例（negative_passages）
                if "negative_passages" in item:
                    if query_id not in qrels:
                        qrels[query_id] = []
                    
                    for passage in item["negative_passages"]:
                        doc_id = str(passage["docid"])
                        qrels[query_id].append((doc_id, 0))  # 相关性为0
            
            print(f"已加载 {lang} 数据集 ({split}): {len(queries)} 查询, {len(documents)} 文档, {len(qrels)} 查询有相关性判断")
            
            # 保存到本地
            save_miracl_data_to_local(lang, split, queries, documents, qrels)
            
            return queries, documents, qrels
            
        except Exception as e:
            print(f"从Hugging Face加载 {lang} 数据集 ({split}) 时出错: {e}")
            raise
    
    # 从本地文件加载数据
    try:
        # 加载查询
        queries = {}
        with open(base_path / "queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = data["id"]
                query_text = data["text"]
                queries[query_id] = query_text

        # 加载文档
        documents = {}
        with open(base_path / "corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                doc_id = data["id"]
                # 使用text或者如果有title则组合title和text
                doc_text = data["text"]
                if "title" in data and data["title"]:
                    doc_text = data["title"] + " " + doc_text
                documents[doc_id] = doc_text

        # 加载相关性判断
        qrels = {}
        with open(base_path / "qrels.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = data["qid"]
                doc_id = data["docid"]
                relevance = data["relevance"]
                
                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append((doc_id, int(relevance)))

        print(f"已从本地加载 {lang} ({split}) 数据集: {len(queries)} 查询, {len(documents)} 文档, {len(qrels)} 查询有相关性判断")
        return queries, documents, qrels
    except Exception as e:
        print(f"从本地加载 {lang} ({split}) 数据集时出错: {e}，将尝试从Hugging Face加载")
        return load_miracl_data(lang, split, force_hf=True)

def save_miracl_data_to_local(lang: str, split: str, queries: Dict[str, str], documents: Dict[str, str], qrels: Dict[str, List[Tuple[str, int]]]):
    """保存数据到本地JSONL文件"""
    base_path = Path(f"datasets/miracl/{lang}/{split}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 保存查询
    with open(base_path / "queries.jsonl", "w", encoding="utf-8") as f:
        for query_id, query_text in queries.items():
            data = {"id": query_id, "text": query_text}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # 保存文档
    with open(base_path / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id, doc_text in documents.items():
            data = {"id": doc_id, "title": "", "text": doc_text}
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    # 保存相关性判断
    with open(base_path / "qrels.jsonl", "w", encoding="utf-8") as f:
        for query_id, doc_list in qrels.items():
            for doc_id, rel in doc_list:
                data = {"qid": query_id, "docid": doc_id, "relevance": rel}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"已将 {lang} 数据集 ({split}) 保存到本地: {base_path}")

def create_training_data(queries: Dict[str, str], documents: Dict[str, str], qrels: Dict[str, List[Tuple[str, int]]], num_negatives: int = None) -> pd.DataFrame:
    """
    创建训练数据，为每个查询创建正负样本对
    
    Args:
        queries: 查询字典
        documents: 文档字典
        qrels: 相关性字典
        num_negatives: 每个查询要采样的负样本数量
                      0表示不添加负样本
                      None表示使用所有负样本
                      >0表示使用指定数量的负样本
        
    Returns:
        训练数据DataFrame
    """
    data = []
    # 为每个查询创建样本对
    for query_id, query_text in tqdm(queries.items(), desc="创建数据"):
        if query_id not in qrels:
            continue
            
        # 获取该查询的所有相关文档ID和相关性得分
        relevant_docs_dict = {doc_id: rel for doc_id, rel in qrels[query_id]}
        relevant_doc_ids = set(relevant_docs_dict.keys())
        
        # 添加所有相关文档（正样本）
        for doc_id in relevant_doc_ids:
            if doc_id in documents:  # 确保文档存在
                data.append({
                    'sample1': query_text,
                    'sample2': documents[doc_id],
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance': relevant_docs_dict[doc_id]
                })
        
        # 处理负样本逻辑
        if num_negatives == 0:
            # 如果num_negatives为0，不添加负样本
            continue
            
        # 获取所有不相关文档ID（负样本候选）
        irrelevant_doc_ids = [doc_id for doc_id in documents.keys() if doc_id not in relevant_doc_ids]
        
        # 根据参数决定使用多少负样本
        if num_negatives is None:
            # 如果num_negatives为None，使用所有负样本
            selected_irrelevant_docs = irrelevant_doc_ids
        elif num_negatives > 0 and len(irrelevant_doc_ids) > num_negatives:
            # 如果指定了负样本数量，随机选择指定数量的负样本
            selected_irrelevant_docs = random.sample(irrelevant_doc_ids, num_negatives)
        else:
            # 其他情况使用所有可用的负样本
            selected_irrelevant_docs = irrelevant_doc_ids
        
        # 添加选定的负样本
        for doc_id in selected_irrelevant_docs:
            data.append({
                'sample1': query_text,
                'sample2': documents[doc_id],
                'query_id': query_id,
                'doc_id': doc_id,
                'relevance': 0
            })

    df = pd.DataFrame(data)
    relevant_count = sum(df['relevance'] > 0)
    irrelevant_count = len(df) - relevant_count
    print(f"创建了 {len(df)} 个样本，其中包含 {relevant_count} 个相关文档样本和 {irrelevant_count} 个不相关文档样本")
    return df

def process_miracl_dataset(lang_list: List[str] = None, num_negatives: int = None, include_train: bool = False, force_hf: bool = False) -> Dict[str, pd.DataFrame]:
    """
    处理指定语言的MIRACL数据集
    
    Args:
        lang_list: 要处理的语言列表，None表示处理所有语言
        num_negatives: 每个查询采样的负样本数量
                      0表示不添加负样本
                      None表示使用所有负样本
                      >0表示使用指定数量的负样本
        include_train: 是否包含训练集
        force_hf: 是否强制从Hugging Face加载
        
    Returns:
        处理后的数据集字典 {语言: DataFrame}
    """
    # 如果未指定语言，则使用所有支持的语言
    if lang_list is None:
        lang_list = MIRACL_LANGUAGES
    elif isinstance(lang_list, str):
        lang_list = [lang_list]
    
    all_data = {}
    for lang in lang_list:
        try:
            print(f"\n开始处理 {lang} 语言的MIRACL数据...")
            
            # 加载dev数据
            queries, documents, qrels = load_miracl_data(lang, "dev", force_hf=force_hf)
            dev_data = create_training_data(queries, documents, qrels, num_negatives=num_negatives)
            
            # 保存处理后的dev数据
            dev_dir = Path(f"datasets/miracl/{lang}/dev")
            dev_dir.mkdir(parents=True, exist_ok=True)
            dev_data.to_pickle(dev_dir / "processed_data.pkl")
            
            # 如果需要处理训练集
            if include_train:
                train_queries, train_documents, train_qrels = load_miracl_data(lang, "train", force_hf=force_hf)
                train_data = create_training_data(train_queries, train_documents, train_qrels, num_negatives=num_negatives)
                
                # 保存处理后的train数据
                train_dir = Path(f"datasets/miracl/{lang}/train")
                train_dir.mkdir(parents=True, exist_ok=True)
                train_data.to_pickle(train_dir / "processed_data.pkl")
                
                # 合并dev和train数据
                lang_data = pd.concat([dev_data, train_data], ignore_index=True)
                print(f"已合并 {lang} 语言的dev数据 ({len(dev_data)} 条) 和 train数据 ({len(train_data)} 条)")
            else:
                lang_data = dev_data
            
            all_data[lang] = lang_data
        except Exception as e:
            print(f"处理 {lang} 数据集时出错: {e}")
            continue
    
    # 如果只处理一种语言，直接返回该语言的数据
    if len(lang_list) == 1:
        return all_data.get(lang_list[0])
    
    # 返回所有语言数据的字典
    return all_data

def load_train_dev_data(lang: str, train_sample_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练和测试数据
    
    Args:
        lang: 语言代码
        train_sample_ratio: 从完整训练集中采样的比例，默认为1.0表示使用全部数据
        
    Returns:
        train_data, test_data
    """
    print(f"加载 {lang} 语言的训练和测试数据...")
    
    # 检查处理好的数据是否存在，不存在则处理
    dev_data_path = Path(f"datasets/miracl/{lang}/dev/processed_data.pkl")
    if not dev_data_path.exists():
        process_miracl_dataset(lang, include_train=True)
    
    # 加载测试集(dev数据)
    test_data = pd.read_pickle(dev_data_path)
    
    # 加载训练集
    train_data_path = Path(f"datasets/miracl/{lang}/train/processed_data.pkl")
    full_train_data = pd.read_pickle(train_data_path)
    
    # 从训练集中采样(如果比例小于1)
    if train_sample_ratio < 1.0:
        train_size = max(1, int(len(full_train_data) * train_sample_ratio))
        train_data = full_train_data.sample(n=train_size, random_state=42)
    else:
        train_data = full_train_data
    
    print(f"数据准备完成: 训练集 {len(train_data)} 样本, 测试集 {len(test_data)} 样本")
    
    return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description="处理MIRACL数据集")
    parser.add_argument("--langs", nargs="+", default=None, help="要处理的语言列表，如ar bn")
    parser.add_argument("--force_hf", action="store_true", default=False, help="强制使用Hugging Face加载数据集")
    parser.add_argument("--num_negatives", type=int, default=0, help="每个查询要生成的负样本数量，0表示不添加负样本，None表示使用全部负样本")
    parser.add_argument("--include_train", action="store_true", default=False, help="是否同时处理训练集")
    args = parser.parse_args()
    
    process_miracl_dataset(
        args.langs, 
        num_negatives=args.num_negatives, 
        include_train=args.include_train,
        force_hf=args.force_hf
    )

if __name__ == "__main__":
    main()

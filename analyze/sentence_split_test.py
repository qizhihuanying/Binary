#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
import argparse
import os
import platform
from typing import List, Dict, Tuple, Union

# 设置中文字体支持
def set_chinese_font():
    system = platform.system()
    if system == 'Windows':
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        except:
            plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system == 'Linux':
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']

    
    plt.rcParams['axes.unicode_minus'] = False  

def load_data(file_path: str, max_samples: int = None) -> List[str]:
    """加载JSONL文件中的文本数据"""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                sentences.append(data['text'])
                if max_samples and len(sentences) >= max_samples:
                    break
    return sentences

def binarize_vector(vector: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """将向量二值化，大于阈值的设为1，小于等于阈值的设为0"""
    return (vector > threshold).astype(np.float32)

def split_sentence(sentence: str, n_segments: int = 2) -> List[str]:
    """将句子分成n段"""
    chars = list(sentence)
    total_length = len(chars)
    segment_length = total_length // n_segments
    
    segments = []
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < n_segments - 1 else total_length
        segments.append(''.join(chars[start:end]))
    
    return segments

def calculate_metrics(
    model,
    sentences: List[str],
    split_sentences: List[List[str]],
    use_binary: bool = False,
    threshold: float = 0.0,
    batch_size: int = 32
) -> Dict[str, List[float]]:
    """计算原始句子与分割句子的向量相似度"""
    results = {
        'cosine_sim': [],
        'euclidean_dist': [],
        'sentence_lens': []
    }
    
    # 使用批处理进行计算
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]
        batch_splits = split_sentences[i:i+batch_size]
        
        # 获取分段数量
        n_segments = len(batch_splits[0]) if batch_splits else 0
        
        if n_segments == 0:
            continue
        
        # 准备批量编码的文本
        all_texts = batch_sentences.copy()
        segment_texts_list = [[] for _ in range(n_segments)]
        
        # 收集每个位置的分段
        for splits in batch_splits:
            for j, segment in enumerate(splits):
                segment_texts_list[j].append(segment)
        
        # 将所有文本添加到编码列表中
        for segment_texts in segment_texts_list:
            all_texts.extend(segment_texts)
        
        # 批量编码
        all_vecs = model.encode(all_texts)
        
        # 分离向量
        orig_vecs = all_vecs[:len(batch_sentences)]
        segment_vecs_list = []
        
        offset = len(batch_sentences)
        for j in range(n_segments):
            segment_vecs = all_vecs[offset:offset+len(batch_sentences)]
            segment_vecs_list.append(segment_vecs)
            offset += len(batch_sentences)
        
        # 如果需要二值化向量
        if use_binary:
            orig_vecs = np.array([binarize_vector(vec, threshold) for vec in orig_vecs])
            for j in range(n_segments):
                segment_vecs_list[j] = np.array([binarize_vector(vec, threshold) for vec in segment_vecs_list[j]])
        
        # 计算每个分段的余弦相似度并取平均
        cos_sims_all = np.zeros(len(batch_sentences))
        for segment_vecs in segment_vecs_list:
            cos_sims = np.diag(cosine_similarity(orig_vecs, segment_vecs))
            cos_sims_all += cos_sims
        
        cos_sims_avg = cos_sims_all / n_segments
        
        # 计算每个分段的欧氏距离并取平均
        euc_dists_all = np.zeros(len(batch_sentences))
        for segment_vecs in segment_vecs_list:
            euc_dists = np.array([np.linalg.norm(orig - segment) for orig, segment in zip(orig_vecs, segment_vecs)])
            euc_dists_all += euc_dists
        
        euc_dists_avg = euc_dists_all / n_segments
        
        # 存储结果
        results['cosine_sim'].extend(cos_sims_avg.tolist())
        results['euclidean_dist'].extend(euc_dists_avg.tolist())
        results['sentence_lens'].extend([len(s) for s in batch_sentences])
        
    return results

def main():
    # 设置中文字体
    set_chinese_font()
    
    parser = argparse.ArgumentParser(description='测试句子分割后的向量表示')
    parser.add_argument('--lang', type=str, default='zh', help='语言代码')
    parser.add_argument('--model', type=str, default='intfloat/multilingual-e5-base', help='要使用的模型名称')
    parser.add_argument('--samples', type=int, default=12029, help='要处理的样本数量')
    parser.add_argument('--threshold', type=float, default=0.0, help='二值化阈值')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--segments', type=int, default=2, help='将句子分割成多少段')
    args = parser.parse_args()
    
    # 设置数据路径
    data_path = f'datasets/miracl/{args.lang}/train/corpus.jsonl'
    if not os.path.exists(data_path):
        print(f"数据文件 {data_path} 不存在!")
        return
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = SentenceTransformer(args.model)
    
    # 加载数据
    print(f"从 {data_path} 加载数据")
    sentences = load_data(data_path, args.samples)
    
    # 分割句子成指定的段数
    print(f"将句子分割成 {args.segments} 段...")
    split_sentences = [split_sentence(s, args.segments) for s in sentences]
    
    # 计算浮点数向量的相似度
    print("计算浮点数向量的相似度...")
    float_results = calculate_metrics(
        model, sentences, split_sentences, use_binary=False, batch_size=args.batch_size
    )
    
    # 计算二值化向量的相似度
    print(f"计算二值化向量的相似度 (阈值 = {args.threshold})...")
    binary_results = calculate_metrics(
        model, sentences, split_sentences, use_binary=True, threshold=args.threshold, batch_size=args.batch_size
    )
    
    # 打印结果
    float_cos_mean = np.mean(float_results['cosine_sim'])
    binary_cos_mean = np.mean(binary_results['cosine_sim'])
    
    float_euc_mean = np.mean(float_results['euclidean_dist'])
    binary_euc_mean = np.mean(binary_results['euclidean_dist'])
    
    print("\n结果汇总:")
    print(f"分段数: {args.segments}")
    print(f"浮点数向量 - 平均余弦相似度: {float_cos_mean:.4f}")
    print(f"二值化向量 - 平均余弦相似度: {binary_cos_mean:.4f}")
    print(f"浮点数向量 - 平均欧氏距离: {float_euc_mean:.4f}")
    print(f"二值化向量 - 平均欧氏距离: {binary_euc_mean:.4f}")
    
    # 比较哪个差异更大
    cos_diff = float_cos_mean - binary_cos_mean
    euc_diff = binary_euc_mean - float_euc_mean  # 注意：欧氏距离越大表示差异越大
    
    print("\n差异比较:")
    if cos_diff > 0:
        print(f"基于余弦相似度: 二值化向量的差异更大 (差值: {abs(cos_diff):.4f})")
    else:
        print(f"基于余弦相似度: 浮点数向量的差异更大 (差值: {abs(cos_diff):.4f})")
        
    if euc_diff > 0:
        print(f"基于欧氏距离: 二值化向量的差异更大 (差值: {abs(euc_diff):.4f})")
    else:
        print(f"基于欧氏距离: 浮点数向量的差异更大 (差值: {abs(euc_diff):.4f})")
    
    # 图表标题添加分段信息
    segments_info = f"(分段数: {args.segments})"
    
    # 绘制可视化图表
    plt.figure(figsize=(14, 12))
    
    # 余弦相似度比较
    plt.subplot(2, 2, 1)
    plt.hist(float_results['cosine_sim'], alpha=0.5, label='浮点数向量')
    plt.hist(binary_results['cosine_sim'], alpha=0.5, label='二值化向量')
    plt.xlabel('余弦相似度')
    plt.ylabel('频率')
    plt.title(f'余弦相似度分布比较 {segments_info}')
    plt.legend()
    
    # 欧氏距离比较
    plt.subplot(2, 2, 2)
    plt.hist(float_results['euclidean_dist'], alpha=0.5, label='浮点数向量')
    plt.hist(binary_results['euclidean_dist'], alpha=0.5, label='二值化向量')
    plt.xlabel('欧氏距离')
    plt.ylabel('频率')
    plt.title(f'欧氏距离分布比较 {segments_info}')
    plt.legend()
    
    # 散点图：句子长度与余弦相似度的关系
    plt.subplot(2, 2, 3)
    plt.scatter(float_results['sentence_lens'], float_results['cosine_sim'], 
                alpha=0.5, label='浮点数向量')
    plt.scatter(binary_results['sentence_lens'], binary_results['cosine_sim'], 
                alpha=0.5, label='二值化向量')
    plt.xlabel('句子长度')
    plt.ylabel('余弦相似度')
    plt.title(f'句子长度与余弦相似度的关系 {segments_info}')
    plt.legend()
    
    # 散点图：句子长度与欧氏距离的关系
    plt.subplot(2, 2, 4)
    plt.scatter(float_results['sentence_lens'], float_results['euclidean_dist'], 
                alpha=0.5, label='浮点数向量')
    plt.scatter(binary_results['sentence_lens'], binary_results['euclidean_dist'], 
                alpha=0.5, label='二值化向量')
    plt.xlabel('句子长度')
    plt.ylabel('欧氏距离')
    plt.title(f'句子长度与欧氏距离的关系 {segments_info}')
    plt.legend()
    
    plt.tight_layout()
    output_file = f'analyze/vector_comparison_results_segments{args.segments}.png'
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"\n结果图表已保存为 '{output_file}'")

if __name__ == "__main__":
    main() 
import argparse
from pathlib import Path
import pandas as pd
import torch
from functools import partial
from transformers import AutoModel, AutoTokenizer
import os
import json

from _models.huggingface.huggingface import get_device
from _models.model import get_embedding_func_batched
from trainer import train


MIRACL_LANGUAGES = [
    "ar", "bn", "en", "es", "fi", "fr", "hi", "id", 
    "ja", "ko", "ru", "sw", "te", "th", "zh", "de", "yo", "fa"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate binary encoder")
    parser.add_argument("--local_model_names", type=str, nargs="+", default=["intfloat/multilingual-e5-base"], help="List of local model names")
    parser.add_argument("--api_model_names", type=str, nargs="+", default=[], help="List of API model names")
    parser.add_argument("--output_dir", type=str, default="project/models/binary_head", help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=0, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--test_ratio", type=float, default=1.0, help="Test set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Validation set ratio except test set")
    parser.add_argument("--base_trainable_layers", type=int, default=0, help="Number of trainable layers in base model")
    parser.add_argument("--use_binary_head", action="store_true", default=False, help="Whether to use binary head")
    parser.add_argument("--dataset", type=str, default="miracl", help="Dataset name, currently only 'miracl' is supported")
    parser.add_argument("--langs", type=str, nargs="+", default=["zh"], help="Languages to process for MIRACL dataset (e.g., ar bn)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--train_sample_ratio", type=float, default=0.2, help="Ratio of training data to use from train split")
    
    return parser.parse_args()


def load_local_models(model_names, device):
    models = []
    tokenizers = []
    
    for model_name in model_names:
        print(f"加载本地模型: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        models.append(model)
        tokenizers.append(tokenizer)
    
    return models, tokenizers


def prepare_api_embedding_funcs(model_names):
    embedding_funcs = []
    
    for model_name in model_names:
        print(f"准备API模型: {model_name}")
        embedding_funcs.append(partial(get_embedding_func_batched(model_name)))
    
    return embedding_funcs


def prepare_data(args):
    # 处理MIRACL数据集
    if args.dataset == "miracl":
        # 如果指定了语言
        if args.langs and len(args.langs) > 0:
            print(f"准备 {args.langs} 语言的数据...")
            
            # 如果只有一种语言，使用优化的数据加载方式
            if len(args.langs) == 1:
                lang = args.langs[0]
                
                # 加载train和dev拆分的数据
                train_sample_ratio = args.train_sample_ratio
                print(f"从train拆分中选择 {train_sample_ratio*100}% 的数据用于训练")
                
                # 直接加载处理好的数据文件
                # 检查测试数据文件是否存在
                dev_data_path = Path(f"datasets/miracl/{lang}/dev/processed_data.pkl")
                if not dev_data_path.exists():
                    raise FileNotFoundError(f"找不到处理好的{lang}测试数据文件: {dev_data_path}，请先运行make_dataset.py处理数据")
                
                # 检查训练数据文件是否存在
                train_data_path = Path(f"datasets/miracl/{lang}/train/processed_data.pkl")
                if not train_data_path.exists():
                    raise FileNotFoundError(f"找不到处理好的{lang}训练数据文件: {train_data_path}，请先运行make_dataset.py处理数据")
                
                print(f"加载处理好的{lang}语言数据文件...")
                
                # 加载测试集(dev数据)
                test_data = pd.read_pickle(dev_data_path)
                
                # 加载完整训练集
                full_train_data = pd.read_pickle(train_data_path)
                
                # 从训练集中采样(如果比例小于1)
                if train_sample_ratio < 1.0:
                    train_size = max(1, int(len(full_train_data) * train_sample_ratio))
                    train_data = full_train_data.sample(n=train_size, random_state=42)
                else:
                    train_data = full_train_data
                
                # 根据val_ratio动态拆分训练集和验证集
                if args.val_ratio > 0:
                    # 计算验证集大小
                    val_size = int(len(train_data) * args.val_ratio)
                    val_size = max(1, val_size)  # 确保至少有1个样本
                    
                    # 拆分验证集
                    val_data = train_data.sample(n=val_size, random_state=42)
                    train_data = train_data.drop(val_data.index)
                    
                    print(f"根据val_ratio={args.val_ratio}拆分: 训练集 {len(train_data)} 样本, 验证集 {len(val_data)} 样本")
                else:
                    # 如果val_ratio为0，则不使用验证集
                    val_data = pd.DataFrame(columns=train_data.columns)
                    print(f"val_ratio=0，不使用验证集")
                
                print(f"数据准备完成: 训练集 {len(train_data)} 样本, 验证集 {len(val_data)} 样本, 测试集 {len(test_data)} 样本")
                return train_data, val_data, test_data
            else:
                # 多语言情况
                data_frames = []
                for lang in args.langs:
                    data_path = Path(f"datasets/miracl/{lang}/dev/processed_data.pkl")
                    if not data_path.exists():
                        raise FileNotFoundError(f"找不到处理好的{lang}语言数据文件: {data_path}，请先运行make_dataset.py处理数据")
                    
                    print(f"加载处理好的MIRACL数据集，语言: {lang}, 文件: {data_path}")
                    lang_data = pd.read_pickle(data_path)
                    data_frames.append(lang_data)
                
                data = pd.concat(data_frames, ignore_index=True)
        else:
            # 使用所有支持的语言
            data_frames = []
            for lang in MIRACL_LANGUAGES:
                data_path = Path(f"datasets/miracl/{lang}/dev/processed_data.pkl")
                if data_path.exists():
                    print(f"加载处理好的MIRACL数据集，语言: {lang}, 文件: {data_path}")
                    lang_data = pd.read_pickle(data_path)
                    data_frames.append(lang_data)
                else:
                    print(f"警告: 找不到{lang}语言的处理好的数据文件: {data_path}")
            
            if not data_frames:
                raise ValueError("没有找到任何处理好的数据文件，请先运行make_dataset.py处理数据")
                
            data = pd.concat(data_frames, ignore_index=True)
            
        if data is None or len(data) == 0:
            raise ValueError("数据集为空或处理失败")
        
        # 如果是单语言使用load_train_dev_data，则已经拆分好数据
        if args.langs and len(args.langs) == 1:
            return train_data, val_data, test_data
            
        # 对于多语言或所有语言的情况，拆分数据集
        total_size = len(data)
        
        # 特殊处理：当test_ratio=1.0时，所有数据都用于测试，训练集和验证集为空
        if args.test_ratio >= 1.0:
            test_data = data
            # 返回空的DataFrame作为训练集和验证集
            empty_df = pd.DataFrame(columns=data.columns)
            return empty_df, empty_df, test_data
        
        # 正常流程：先分出测试集
        test_size = int(total_size * args.test_ratio)
        test_data = data.sample(n=test_size, random_state=42)
        remaining_data = data.drop(test_data.index)
        
        # 从剩余数据中分出验证集（基于剩余数据的比例）
        remaining_size = len(remaining_data)
        val_size = int(remaining_size * args.val_ratio)
        
        # 确保验证集至少有1条数据
        val_size = max(1, val_size)
        
        # 从剩余数据中分出验证集
        val_data = remaining_data.sample(n=val_size, random_state=42)
        train_data = remaining_data.drop(val_data.index)

        return train_data, val_data, test_data
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")


def save_results(results, args):
    if results is None:
        print("没有评估结果可保存")
        return
        
    # 创建结果目录
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取模型名称
    model_names = []
    for model_name in args.local_model_names:
        model_names.append(f"local_{model_name.replace('/', '_')}")
    for model_name in args.api_model_names:
        model_names.append(f"api_{model_name.replace('/', '_')}")
    
    # 创建结果摘要
    summary = {
        "dataset": args.dataset,
        "langs": args.langs,
        "use_binary_head": args.use_binary_head,
        "models": model_names,
        "results": results
    }
    
    # 生成唯一的文件名
    filename = f"{args.dataset}_{'with' if args.use_binary_head else 'without'}_binary_head.json"
    file_path = results_dir / filename
    
    # 保存结果
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"结果已保存到 {file_path}")


def main():
    args = parse_args()
    device = get_device(use_gpu=True)
    print(f"Using device: {device}")

    print("\n======= Parameters =======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=======================\n")

    train_data, val_data, test_data = prepare_data(args)

    models, tokenizers = load_local_models(args.local_model_names, device)
    embedding_funcs = prepare_api_embedding_funcs(args.api_model_names)

    train(
        models=models,
        tokenizers=tokenizers,
        embedding_funcs=embedding_funcs,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        temp=args.temp,
        num_trainable_layers=args.base_trainable_layers,
        output_dir=args.output_dir,
        use_binary_head=args.use_binary_head
    )


if __name__ == "__main__":
    main()
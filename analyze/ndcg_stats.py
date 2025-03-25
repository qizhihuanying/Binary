import os
import re
import glob
import pandas as pd
import argparse
from collections import defaultdict

# 统计目录下所有符合条件的日志文件
def analyze_logs(log_dir='logs/intfloat', file_pattern='*.log', output_file=None):
    # 结果存储 - 使用参数字典作为键
    results = defaultdict(list)
    param_keys = set()  # 存储所有参数名
    
    # 获取所有日志文件
    log_files = glob.glob(os.path.join(log_dir, file_pattern))
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    total_experiments = 0
    
    for log_file in log_files:
        # 从文件名中提取参数
        # 提取文件名，去掉路径和扩展名
        file_name = os.path.basename(log_file)
        file_base = os.path.splitext(file_name)[0]  # 去掉.log扩展名
        
        # 从文件名中提取所有参数
        parts = file_base.split('+')
        params = {}
        
        for part in parts:
            # 查找所有格式为key=value的部分
            if '=' in part:
                key, value = part.split('=', 1)
                params[key] = value
                param_keys.add(key)  # 记录所有的参数名
        
        if params:  # 如果提取到了参数
            # 读取文件内容
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找所有训练实验的记录
            # 通过寻找"开始训练过程"来区分不同的实验
            experiment_blocks = re.split(r'开始训练过程', content)
            
            for block in experiment_blocks:
                if not block.strip():
                    continue
                    
                # 在每个实验块中查找NDCG@10的值
                ndcg_matches = re.findall(r'model_0: (\d+\.\d+)', block)
                if ndcg_matches:
                    # 获取实验的最终NDCG@10值
                    final_ndcg = float(ndcg_matches[-1])
                    # 使用参数的元组作为键
                    key = tuple(sorted([(k, params.get(k, 'N/A')) for k in param_keys]))
                    results[key].append(final_ndcg)
                    total_experiments += 1
                    param_str = ', '.join([f"{k}: {params.get(k, 'N/A')}" for k in sorted(param_keys)])
                    print(f"文件: {log_file}, 参数: {param_str}, NDCG@10: {final_ndcg}")
    
    print(f"\n总共找到 {total_experiments} 个实验结果")
    
    if not results:
        print("未找到任何符合条件的NDCG@10值，请检查日志文件格式")
        return
    
    # 计算每个参数组合的平均值
    data = []
    for params_tuple, values in results.items():
        avg = sum(values) / len(values)
        params_dict = {k: v for k, v in params_tuple}
        
        row = {
            '平均NDCG@10': round(avg, 4),
            '样本数': len(values),
            '所有值': ', '.join([str(round(v, 4)) for v in values])
        }
        # 添加参数到行
        for k, v in params_tuple:
            row[k] = v
        
        data.append(row)
    
    # 找出最高的平均NDCG@10值及其对应的参数
    best_item = max(data, key=lambda x: x['平均NDCG@10'])
    best_ndcg = best_item['平均NDCG@10']
    best_samples = best_item['样本数']
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 对参数列进行排序
    param_columns = list(param_keys)
    param_sort_keys = []
    
    # 为数字类型的参数添加排序键
    for param in param_columns:
        try:
            df[f'排序键_{param}'] = df[param].apply(lambda x: float(x) if x != 'N/A' else float('-inf'))
            param_sort_keys.append(f'排序键_{param}')
        except:
            # 如果转换失败，则说明不是数字参数，按原样排序
            pass
    
    if param_sort_keys:
        df = df.sort_values(param_sort_keys)
        df = df.drop(param_sort_keys, axis=1)
    
    # 调整列顺序，把参数列放在前面
    cols = param_columns + ['平均NDCG@10', '样本数', '所有值']
    df = df[cols]
    
    # 准备输出文件路径
    if output_file:
        txt_file = output_file
        if not txt_file.lower().endswith('.txt'):
            txt_file += '.txt'
    else:
        txt_file = os.path.join('analyze', 'ndcg_stats_full.txt')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    
    # 创建一个字符串，用于存储所有要写入文件的内容
    output_text = []
    
    # 标题
    output_text.append("=" * 80)
    output_text.append(f"NDCG@10 分析结果 - 共 {total_experiments} 个实验")
    output_text.append("=" * 80)
    output_text.append("")
    
    # 统计结果
    output_text.append("统计结果:")
    header = "".join([f"{param:^12}" for param in param_columns]) + f"{'平均NDCG@10':^15}{'样本数':^8}{'所有值':<40}"
    output_text.append(header)
    output_text.append("-" * (12 * len(param_columns) + 15 + 8 + 40))
    
    for item in data:
        row = "".join([f"{item.get(param, 'N/A'):^12}" for param in param_columns])
        row += f"{item['平均NDCG@10']:^15}{item['样本数']:^8}{item['所有值']:<40}"
        output_text.append(row)
    
    # 打印最佳配置
    best_params_str = ", ".join([f"{param}={best_item.get(param, 'N/A')}" for param in param_columns])
    output_text.append("")
    output_text.append(f"最佳参数配置: {best_params_str}, 平均NDCG@10={best_ndcg} (样本数: {best_samples})")
    
    # 在控制台显示
    print("\n统计结果:")
    print(header)
    print("-" * (12 * len(param_columns) + 15 + 8 + 40))
    
    for item in data:
        row = "".join([f"{item.get(param, 'N/A'):^12}" for param in param_columns])
        row += f"{item['平均NDCG@10']:^15}{item['样本数']:^8}{item['所有值']:<40}"
        print(row)
    
    # 打印最佳配置
    print(f"\n最佳参数配置: {best_params_str}, 平均NDCG@10={best_ndcg} (样本数: {best_samples})")
    
    # 如果有多个参数，对每个参数单独分析最佳其他参数组合
    if len(param_columns) > 1:
        for target_param in param_columns:
            other_params = [p for p in param_columns if p != target_param]
            
            # 添加到输出文本
            output_text.append("")
            output_text.append("=" * 60)
            output_text.append(f"按{target_param}分组的最佳其他参数值:")
            header = f"{target_param:^12}" + "".join([f"{param:^12}" for param in other_params]) + f"{'最高NDCG@10':^15}"
            output_text.append(header)
            output_text.append("-" * (12 * len(param_columns) + 15))
            
            # 控制台输出
            print(f"\n\n按{target_param}分组的最佳其他参数值:")
            print(header)
            print("-" * (12 * len(param_columns) + 15))
            
            # 为每个参数值找出最佳组合
            param_groups = {}
            for item in data:
                param_value = item[target_param]
                ndcg = item['平均NDCG@10']
                
                if param_value not in param_groups or ndcg > param_groups[param_value]['ndcg']:
                    param_groups[param_value] = {
                        'ndcg': ndcg,
                        'params': {p: item.get(p, 'N/A') for p in other_params}
                    }
            
            # 尝试按数值排序，如果失败则按字符串排序
            try:
                sorted_param_values = sorted(param_groups.keys(), key=float)
            except:
                sorted_param_values = sorted(param_groups.keys())
                
            for param_value in sorted_param_values:
                group = param_groups[param_value]
                
                # 格式化输出行
                row_txt = f"{param_value:^12}"
                row_txt += "".join([f"{group['params'].get(p, 'N/A'):^12}" for p in other_params])
                row_txt += f"{group['ndcg']:^15}"
                
                # 添加到输出文本
                output_text.append(row_txt)
                
                # 打印到控制台
                print(row_txt)
    
    # 保存为TXT文件
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_text))
    print(f"\n所有结果已保存至文本文件: {txt_file}")

def main():
    parser = argparse.ArgumentParser(description="分析日志文件中的NDCG@10值")
    parser.add_argument("--log_dir", type=str, default="logs/multilingual-e5-base", 
                        help="日志文件目录路径")
    parser.add_argument("--pattern", type=str, default="*.log", 
                        help="日志文件匹配模式，默认为'*.log'")
    parser.add_argument("--output", type=str, default="analyze/ndcg_stats_full.txt", 
                        help="输出文本文件路径，默认为'analyze/ndcg_stats_full.txt'")
    args = parser.parse_args()
    
    analyze_logs(args.log_dir, args.pattern, args.output)

if __name__ == "__main__":
    main() 
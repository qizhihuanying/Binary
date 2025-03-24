import os
import re
import glob
import pandas as pd
from collections import defaultdict

# 统计目录下所有符合条件的日志文件
def analyze_logs():
    # 结果存储 - 使用(lr, l2)元组作为键
    results = defaultdict(list)
    
    # 获取所有日志文件
    log_files = glob.glob('logs/intfloat/*.log')
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    total_experiments = 0
    
    for log_file in log_files:
        # 从文件名中提取参数
        # 提取文件名，去掉路径和扩展名
        file_name = os.path.basename(log_file)
        file_base = os.path.splitext(file_name)[0]  # 去掉.log扩展名
        
        # 从文件名中提取参数
        parts = file_base.split('+')
        lr = None
        l2 = None
        
        for part in parts:
            if part.startswith('lr='):
                lr = part[3:]  # 取'lr='后面的部分
            elif part.startswith('l2='):
                l2 = part[3:]  # 取'l2='后面的部分
        
        if lr and l2:
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
                    # 使用(lr, l2)元组作为键
                    key = (lr, l2)
                    results[key].append(final_ndcg)
                    total_experiments += 1
                    print(f"文件: {log_file}, 学习率: {lr}, L2正则化: {l2}, NDCG@10: {final_ndcg}")
    
    print(f"\n总共找到 {total_experiments} 个实验结果")
    
    if not results:
        print("未找到任何符合条件的NDCG@10值，请检查日志文件格式")
        return
    
    # 计算每个(lr, l2)组合的平均值
    data = []
    for (lr, l2), values in results.items():
        avg = sum(values) / len(values)
        data.append({
            '学习率': lr,
            'L2正则化': l2,
            '平均NDCG@10': round(avg, 4),
            '样本数': len(values),
            '所有值': ', '.join([str(round(v, 4)) for v in values])
        })
    
    # 找出最高的平均NDCG@10值及其对应的参数
    best_item = max(data, key=lambda x: x['平均NDCG@10'])
    best_lr = best_item['学习率']
    best_l2 = best_item['L2正则化']
    best_ndcg = best_item['平均NDCG@10']
    best_samples = best_item['样本数']
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 将学习率和L2转换为浮点数进行排序
    df['排序键_lr'] = df['学习率'].apply(lambda x: float(x))
    df['排序键_l2'] = df['L2正则化'].apply(lambda x: float(x))
    df = df.sort_values(['排序键_lr', '排序键_l2'])
    df = df.drop(['排序键_lr', '排序键_l2'], axis=1)
    
    # 添加一个空行和一个总结行
    summary_df = pd.DataFrame([
        {
            '学习率': '',
            'L2正则化': '',
            '平均NDCG@10': '',
            '样本数': '',
            '所有值': ''
        },
        {
            '学习率': f'最佳配置: {best_lr}',
            'L2正则化': best_l2,
            '平均NDCG@10': best_ndcg,
            '样本数': best_samples,
            '所有值': f'最高NDCG@10值: {best_ndcg}'
        }
    ])
    
    # 合并原始数据和总结数据
    final_df = pd.concat([df, summary_df], ignore_index=True)
    
    # 保存为CSV
    csv_file = 'analyze/ndcg_stats_full.csv'
    final_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {csv_file}")
    
    # 手动格式化输出结果
    print("\n统计结果:")
    print(f"{'学习率':^10}{'L2正则化':^12}{'平均NDCG@10':^15}{'样本数':^8}{'所有值':<40}")
    print("-" * 85)
    
    # 按学习率和l2值排序
    sorted_data = sorted(data, key=lambda x: (float(x['学习率']), float(x['L2正则化'])))
    
    for item in sorted_data:
        print(f"{item['学习率']:^10}{item['L2正则化']:^12}{item['平均NDCG@10']:^15}{item['样本数']:^8}{item['所有值']:<40}")
    
    # 打印最佳配置
    print(f"\n最佳参数配置: 学习率={best_lr}, L2正则化={best_l2}, 平均NDCG@10={best_ndcg} (样本数: {best_samples})")
    
    # 按学习率分组显示最佳L2值
    print("\n\n按学习率分组的最佳L2值:")
    print(f"{'学习率':^10}{'最佳L2':^12}{'最高NDCG@10':^15}")
    print("-" * 40)
    
    # 为每个学习率找出最佳的L2值
    lr_groups = {}
    for item in data:
        lr = item['学习率']
        l2 = item['L2正则化']
        ndcg = item['平均NDCG@10']
        
        if lr not in lr_groups or ndcg > lr_groups[lr][1]:
            lr_groups[lr] = (l2, ndcg)
    
    # 按学习率排序显示最佳L2值
    for lr, (best_l2, best_ndcg) in sorted(lr_groups.items(), key=lambda x: float(x[0])):
        print(f"{lr:^10}{best_l2:^12}{best_ndcg:^15}")

if __name__ == "__main__":
    analyze_logs() 
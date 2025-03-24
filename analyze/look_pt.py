import torch
import sys

def view_pt_file(file_path):
    try:
        # 加载 pt 文件内容，使用 CPU 加载，避免 GPU 环境限制
        data = torch.load(file_path, map_location=torch.device('cpu'))
    except Exception as e:
        print("加载 pt 文件时出错:", e)
        return

    # 输出数据的基本信息
    print("数据类型:", type(data))
    
    # 如果数据是字典，则逐键打印
    if isinstance(data, dict):
        print("字典键:")
        for key, value in data.items():
            print(f"  键: {key}，类型: {type(value)}")
            # 如果值是 tensor，显示其形状和部分数据
            if torch.is_tensor(value):
                print(f"    张量形状: {value.size()}")
                # 显示前几个数值
                print("    数值预览:", value.flatten()[:10])
            else:
                print("    内容预览:", value)
    else:
        # 如果数据不是字典，则直接打印
        print("数据内容:")
        print(data)

def main():
    if len(sys.argv) != 2:
        print("用法: python view_pt.py <file.pt>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    view_pt_file(file_path)

if __name__ == "__main__":
    main()

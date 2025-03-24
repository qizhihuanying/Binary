import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryHead(nn.Module):
    def __init__(self, unified_dim=512, output_dim=256, temp=1.0, use_binary_head=True):
        """
        Args:
            unified_dim: 统一的中间维度，所有输入都会先映射到这个维度
            output_dim: 最终的二值化输出维度
            temp: 已不再使用的参数，保留只是为了接口兼容
            use_binary_head: 是否使用二值化头
        """
        super().__init__()
        self.unified_dim = unified_dim
        self.output_dim = output_dim
        self.temp = 1.0  # 不再使用
        self.use_binary_head = use_binary_head
        self.training = True
        
        # 维度统一层字典
        self.dim_unifiers = nn.ModuleDict()

        self.binary_projector = nn.Sequential(
            nn.Linear(unified_dim, unified_dim // 2),
            nn.BatchNorm1d(unified_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(unified_dim // 2, output_dim),
            nn.BatchNorm1d(unified_dim // 2),
            nn.LeakyReLU(0.2),
        )
        
    def get_dim_unifier(self, input_dim):
        dim_key = str(input_dim)
        if dim_key not in self.dim_unifiers:
            unifier = nn.Sequential(
                nn.Linear(input_dim, self.unified_dim),
                nn.LayerNorm(self.unified_dim)
            )
            unifier = unifier.to(next(self.binary_projector.parameters()).device)
            self.dim_unifiers[dim_key] = unifier
        return self.dim_unifiers[dim_key]

    class BinarySTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # 保存输入用于反向传播
            ctx.save_for_backward(input)
            # 二值化为0和1
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            # 只对[-1, 1]范围内的值传递梯度
            grad_input = grad_output.clone()
            # 对于太小或太大的值，梯度设为0
            grad_input = grad_input * (torch.abs(input) <= 1).float()
            return grad_input

    def forward(self, x):
        input_dim = x.size(-1)
        
        # 如果不使用二值化头，直接返回原始输入
        if not self.use_binary_head:
            # 确保返回的张量有梯度
            if not x.requires_grad:
                x = x.detach().clone().requires_grad_(True)
            return x
            
        dim_unifier = self.get_dim_unifier(input_dim)
        x = dim_unifier(x)
        
        # 应用完整的处理流程
        x = self.binary_projector(x)
        
        if self.training:
            # 训练时使用STE进行二值化
            return self.BinarySTE.apply(x)
        else:
            # 推理时使用硬二值化
            return (x > 0).float()
    
    def save_model(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / "binary_head_full.pt"
        state = {
            'unified_dim': self.unified_dim,
            'output_dim': self.output_dim,
            'use_binary_head': self.use_binary_head,
            'dim_unifiers': {str(k): v.state_dict() for k, v in self.dim_unifiers.items()},
            'binary_projector': self.binary_projector.state_dict(),
            'supported_dims': [str(k) for k in self.dim_unifiers.keys()] 
        }
        torch.save(state, model_path)

        config = {
            "unified_dim": self.unified_dim,
            "output_dim": self.output_dim,
            "use_binary_head": self.use_binary_head,
            "supported_dims": list(self.dim_unifiers.keys())
        }
        config_path = output_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Model saved to: {model_path}")
        print(f"Config saved to: {config_path}")
        print(f"Supported input dimensions: {config['supported_dims']}")
        
    @classmethod
    def load_model(cls, path, device):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        try:
            state = torch.load(path, map_location=device)
            
            # 创建模型实例
            model = cls(
                unified_dim=state['unified_dim'],
                output_dim=state['output_dim'],
                use_binary_head=state.get('use_binary_head', True)
            ).to(device)
            
            # 加载维度统一层和二值投影层的权重
            for dim_key, unifier_state in state['dim_unifiers'].items():
                input_dim = int(float(dim_key))
                unifier = model.get_dim_unifier(input_dim)
                unifier.load_state_dict(unifier_state)
            model.binary_projector.load_state_dict(state['binary_projector'])
            
            print(f"Loaded model supports input dimensions: {state['supported_dims']}")
            return model
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
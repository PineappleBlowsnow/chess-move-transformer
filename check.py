from src.model import ChessConfig
from src.utils import print_parameter_budget

# 在这里尝试不同的组合
config = ChessConfig(
    vocab_size=1682, # 建议使用你之前成功运行的词表大小
    n_embd=128,      # 宽度
    n_layer=6,       # 深度（增加层数通常能显著提高合法率）
    n_head=8,        # 注意：n_embd 必须能被 n_head 整除
    n_inner=227    # 前馈网络的隐藏层大小
)

print("-" * 30)
print("当前配置参数统计：")
print_parameter_budget(config)
print("-" * 30)
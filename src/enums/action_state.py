from enum import Enum, unique

@unique
class ActionState(Enum):
    """射箭动作状态枚举类"""
    LIFT = "Lift"       # 举弓
    DRAW = "Draw"       # 开弓
    SOLID = "Solid"     # 固势
    RELEASE = "Release" # 撒放
    UNKNOWN = "Unknown"        # 未知状态

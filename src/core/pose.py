import numpy as np

from src.enums.action_state import ActionState

class Pose:
    
    angle_list = []
    release_angle = None
    
    @staticmethod
    def calculate_angle(c, d, a, b) -> float:
        """计算两向量夹角（0-360度）"""
        # 转换为numpy数组
        vec_ab = np.array([b[0]-a[0], b[1]-a[1]])
        vec_cd = np.array([d[0]-c[0], d[1]-c[1]])
        
        # 计算模长
        norm_ab = np.linalg.norm(vec_ab)
        norm_cd = np.linalg.norm(vec_cd)
        
        if norm_ab == 0 or norm_cd == 0:
            return 0.0
            
        # 计算夹角（带方向）
        cos_theta = np.dot(vec_ab, vec_cd) / (norm_ab * norm_cd)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        
        # 判断方向
        cross = np.cross(vec_ab, vec_cd)
        angle_deg = np.degrees(angle_rad)
        return angle_deg if cross >= 0 else 360 - angle_deg

    @classmethod
    def judge_action(cls, angle):
        """
        根据角度判断动作环节
        参数:
            angle (float): 计算出的角度值 (0-360范围)
        """
        cls.angle_list.append(angle)
        
        release_angle_threshold = 4.5  # 固势->撒放 角度骤增差值阈值

        if 330 <= angle < 360 or 0 < angle < 12:
            cls.release_angle = None  # 重置撒放角
            return ActionState.LIFT  # 举弓
        elif 12 <= angle < 150:
            return ActionState.DRAW  # 开弓
        elif cls.release_angle and cls.release_angle - release_angle_threshold <= angle <= 185:
            return ActionState.RELEASE  # 撒放
        elif 150 <= angle < 185:
            previous_angles = cls.angle_list[-4:-1]
            previous_angle = sum(previous_angles) / 3  # 取前三帧的平均值
            if min(previous_angles) >= 150 and 20 > angle - previous_angle >= release_angle_threshold:  # 固势下骤增角度可视为进入撒发环节 (撒放角)
                cls.release_angle = angle
                return ActionState.RELEASE  # 撒放
            return ActionState.SOLID  # 固势
        elif 185 <= angle < 215:
            return ActionState.RELEASE  # 撒放
        else:
            return ActionState.UNKNOWN
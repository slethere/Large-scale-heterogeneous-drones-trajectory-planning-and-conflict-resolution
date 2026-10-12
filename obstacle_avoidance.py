"""
无人机避障算法
- 起飞到50米开始巡航
- 遇到低于120米建筑：改变高度
- 遇到高于120米建筑：改变轨迹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import json

class DroneObstacleAvoidance:
    def __init__(self, order_file, building_file):
        """初始化避障系统"""
        self.orders = pd.read_csv(order_file)
        self.buildings = pd.read_csv(building_file)
        self.cruise_height = 50  # 巡航高度50米
        self.height_threshold = 120  # 高度阈值120米
        self.safety_margin = 20  # 安全裕度（米）
        
    def parse_polygon(self, polygon_str):
        """解析多边形字符串"""
        try:
            # 移除方括号和括号
            polygon_str = polygon_str.strip('[]')
            # 分割顶点
            vertices = []
            for coord in polygon_str.split('),('):
                coord = coord.strip('()')
                lon, lat = map(float, coord.split(','))
                vertices.append((lon, lat))
            return Polygon(vertices)
        except:
            return None
    
    def get_route_bounds(self, start_x, start_y, end_x, end_y, buffer=0.01):
        """获取路径区域范围（增加缓冲区）"""
        min_x = min(start_x, end_x) - buffer
        max_x = max(start_x, end_x) + buffer
        min_y = min(start_y, end_y) - buffer
        max_y = max(start_y, end_y) + buffer
        return min_x, max_x, min_y, max_y
    
    def get_obstacles_in_route(self, start_x, start_y, end_x, end_y):
        """获取路径区域内的障碍物"""
        min_x, max_x, min_y, max_y = self.get_route_bounds(start_x, start_y, end_x, end_y)
        
        obstacles = []
        for idx, building in self.buildings.iterrows():
            # 检查建筑物是否在路径区域内
            if (min_x <= building['longitude'] <= max_x and 
                min_y <= building['latitude'] <= max_y):
                
                polygon = self.parse_polygon(building['polygon_vertices'])
                if polygon:
                    obstacles.append({
                        'id': building['building_id'],
                        'height': building['height'],
                        'polygon': polygon,
                        'center': (building['longitude'], building['latitude'])
                    })
        
        return obstacles
    
    def check_collision(self, point, height, obstacles):
        """检查某点某高度是否与障碍物碰撞"""
        pt = Point(point)
        for obs in obstacles:
            if obs['polygon'].contains(pt) or obs['polygon'].distance(pt) < 0.0001:
                if height < obs['height']:
                    return True, obs
        return False, None
    
    def plan_path_with_avoidance(self, start_x, start_y, end_x, end_y):
        """规划带避障的路径"""
        print(f"\n开始规划路径:")
        print(f"起点: ({start_x:.6f}, {start_y:.6f})")
        print(f"终点: ({end_x:.6f}, {end_y:.6f})")
        
        # 获取路径区域内的障碍物
        obstacles = self.get_obstacles_in_route(start_x, start_y, end_x, end_y)
        print(f"\n路径区域内发现 {len(obstacles)} 个建筑物障碍")
        
        # 分类障碍物
        low_obstacles = [obs for obs in obstacles if obs['height'] < self.height_threshold]
        high_obstacles = [obs for obs in obstacles if obs['height'] >= self.height_threshold]
        
        print(f"- 低于120米建筑: {len(low_obstacles)} 个")
        print(f"- 高于120米建筑: {len(high_obstacles)} 个")
        
        # 生成路径点
        path_3d = self.generate_3d_path(start_x, start_y, end_x, end_y, 
                                        low_obstacles, high_obstacles)
        
        return path_3d, obstacles
    
    def generate_3d_path(self, start_x, start_y, end_x, end_y, 
                         low_obstacles, high_obstacles):
        """生成三维路径"""
        path = []
        
        # 1. 起飞阶段：从0米爬升到50米
        path.append([start_x, start_y, 0, '起飞点'])
        path.append([start_x, start_y, self.cruise_height, '爬升到巡航高度'])
        
        # 2. 计算直线距离和方向
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        num_waypoints = max(50, int(distance * 10000))  # 根据距离确定航点数量
        
        # 当前位置和高度
        current_x = start_x
        current_y = start_y
        current_height = self.cruise_height
        
        # 3. 巡航阶段：逐步向终点移动，实时避障
        for i in range(1, num_waypoints + 1):
            # 计算下一个点的位置
            ratio = i / num_waypoints
            next_x = start_x + (end_x - start_x) * ratio
            next_y = start_y + (end_y - start_y) * ratio
            next_height = current_height
            
            # 检查是否需要避障
            needs_avoidance, obstacle_info = self.check_avoidance_needed(
                next_x, next_y, next_height, low_obstacles, high_obstacles
            )
            
            if needs_avoidance:
                if obstacle_info['type'] == 'low':
                    # 低于120米：提升高度
                    next_height = obstacle_info['height'] + self.safety_margin
                    path.append([next_x, next_y, next_height, 
                                f"提升高度避障(建筑高{obstacle_info['height']:.1f}m)"])
                    current_height = next_height
                else:
                    # 高于120米：绕行
                    detour_points = self.calculate_detour(
                        current_x, current_y, next_x, next_y, 
                        current_height, obstacle_info
                    )
                    for dp in detour_points:
                        path.append(dp)
                    current_x, current_y = detour_points[-1][0], detour_points[-1][1]
                    continue
            else:
                # 无障碍，正常飞行
                path.append([next_x, next_y, next_height, '正常巡航'])
            
            current_x = next_x
            current_y = next_y
        
        # 4. 到达终点上方，保持当前高度
        path.append([end_x, end_y, current_height, '到达终点上方'])
        
        # 5. 降落阶段
        path.append([end_x, end_y, 0, '降落'])
        
        return path
    
    def check_avoidance_needed(self, x, y, height, low_obstacles, high_obstacles):
        """检查是否需要避障"""
        point = Point(x, y)
        
        # 检查低矮建筑物
        for obs in low_obstacles:
            if obs['polygon'].contains(point) or obs['polygon'].distance(point) < 0.0005:
                if height < obs['height'] + self.safety_margin:
                    return True, {
                        'type': 'low',
                        'height': obs['height'],
                        'polygon': obs['polygon']
                    }
        
        # 检查高层建筑物
        for obs in high_obstacles:
            if obs['polygon'].contains(point) or obs['polygon'].distance(point) < 0.0005:
                return True, {
                    'type': 'high',
                    'height': obs['height'],
                    'polygon': obs['polygon'],
                    'center': obs['center']
                }
        
        return False, None
    
    def calculate_detour(self, from_x, from_y, to_x, to_y, height, obstacle_info):
        """计算绕行路径"""
        # 获取建筑物的边界
        bounds = obstacle_info['polygon'].bounds  # (minx, miny, maxx, maxy)
        center_x, center_y = obstacle_info['center']
        
        # 计算绕行方向（顺时针或逆时针）
        # 使用叉积判断目标点在障碍物的哪一侧
        dx = to_x - from_x
        dy = to_y - from_y
        cross = dx * (center_y - from_y) - dy * (center_x - from_x)
        
        # 设置绕行距离
        detour_distance = 0.002  # 约200米
        
        if cross > 0:  # 逆时针绕行
            detour_x = center_x - detour_distance
            detour_y = center_y + detour_distance
        else:  # 顺时针绕行
            detour_x = center_x + detour_distance
            detour_y = center_y - detour_distance
        
        detour_points = [
            [detour_x, detour_y, height, f"绕行高层建筑({obstacle_info['height']:.1f}m)"]
        ]
        
        return detour_points
    
    def visualize_path_3d(self, path_3d, obstacles, order_id):
        """可视化三维路径"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取路径坐标
        path_array = np.array([[p[0], p[1], p[2]] for p in path_3d])
        
        # 绘制路径
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                'b-', linewidth=2, label='无人机路径', marker='o', markersize=3)
        
        # 标记起点和终点
        ax.scatter([path_array[0, 0]], [path_array[0, 1]], [path_array[0, 2]], 
                  c='green', s=200, marker='^', label='起点', zorder=5)
        ax.scatter([path_array[-1, 0]], [path_array[-1, 1]], [path_array[-1, 2]], 
                  c='red', s=200, marker='v', label='终点', zorder=5)
        
        # 绘制建筑物
        for obs in obstacles:
            polygon = obs['polygon']
            coords = list(polygon.exterior.coords)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            
            # 建筑物底部和顶部
            ax.plot(xs, ys, [0]*len(xs), 'gray', alpha=0.3)
            ax.plot(xs, ys, [obs['height']]*len(xs), 'gray', alpha=0.3)
            
            # 建筑物侧面
            for i in range(len(coords)-1):
                ax.plot([xs[i], xs[i]], [ys[i], ys[i]], [0, obs['height']], 
                       'gray', alpha=0.3)
            
            # 根据高度着色
            color = 'orange' if obs['height'] < 120 else 'red'
            alpha = 0.3 if obs['height'] < 120 else 0.5
            
            # 填充建筑物顶部
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts = [list(zip(xs, ys, [obs['height']]*len(xs)))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=alpha, 
                                                facecolor=color, edgecolor='black'))
        
        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [
            plt.Line2D([0], [0], color='b', linewidth=2, label='无人机路径'),
            plt.scatter([], [], c='green', s=100, marker='^', label='起点'),
            plt.scatter([], [], c='red', s=100, marker='v', label='终点'),
            Patch(facecolor='orange', alpha=0.3, label='<120m 建筑'),
            Patch(facecolor='red', alpha=0.5, label='≥120m 建筑')
        ]
        
        ax.set_xlabel('经度 (Longitude)', fontsize=12, fontproperties='SimHei')
        ax.set_ylabel('纬度 (Latitude)', fontsize=12, fontproperties='SimHei')
        ax.set_zlabel('高度 (Height, m)', fontsize=12, fontproperties='SimHei')
        ax.set_title(f'无人机配送避障路径 - 订单 {order_id}', 
                    fontsize=14, fontproperties='SimHei', pad=20)
        ax.legend(handles=legend_elements, loc='upper right', prop={'family': 'SimHei'})
        
        # 设置视角
        ax.view_init(elev=25, azim=45)
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'f:\\PROJECT 2\\dataset\\building\\algrithm\\order_{order_id}_3d_path.png', 
                    dpi=300, bbox_inches='tight')
        print(f"\n3D路径图已保存: order_{order_id}_3d_path.png")
        plt.show()
    
    def process_first_order(self):
        """处理第一个订单"""
        # 获取第一个订单
        order = self.orders.iloc[0]
        order_id = order['task_id']
        start_x = order['sender_X']
        start_y = order['sender_Y']
        end_x = order['receiver_X']
        end_y = order['receiver_Y']
        
        print("="*60)
        print(f"处理订单 {order_id}")
        print("="*60)
        
        # 规划路径
        path_3d, obstacles = self.plan_path_with_avoidance(
            start_x, start_y, end_x, end_y
        )
        
        # 输出路径信息
        print(f"\n生成路径包含 {len(path_3d)} 个航点:")
        print("-"*60)
        for i, waypoint in enumerate(path_3d[:10]):  # 只显示前10个点
            print(f"航点 {i+1}: 经度={waypoint[0]:.6f}, "
                  f"纬度={waypoint[1]:.6f}, 高度={waypoint[2]:.1f}m - {waypoint[3]}")
        if len(path_3d) > 10:
            print(f"... 还有 {len(path_3d)-10} 个航点")
        
        # 保存路径数据
        path_data = {
            'order_id': int(order_id),
            'start': {'longitude': float(start_x), 'latitude': float(start_y)},
            'end': {'longitude': float(end_x), 'latitude': float(end_y)},
            'waypoints': [
                {
                    'longitude': float(wp[0]),
                    'latitude': float(wp[1]),
                    'height': float(wp[2]),
                    'description': wp[3]
                }
                for wp in path_3d
            ],
            'statistics': {
                'total_waypoints': len(path_3d),
                'max_height': max(wp[2] for wp in path_3d),
                'obstacles_in_route': len(obstacles),
                'low_obstacles': len([o for o in obstacles if o['height'] < 120]),
                'high_obstacles': len([o for o in obstacles if o['height'] >= 120])
            }
        }
        
        with open(f'f:\\PROJECT 2\\dataset\\building\\algrithm\\order_{order_id}_path.json', 'w', encoding='utf-8') as f:
            json.dump(path_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n路径数据已保存: order_{order_id}_path.json")
        
        # 可视化
        self.visualize_path_3d(path_3d, obstacles, order_id)
        
        return path_3d, obstacles


if __name__ == "__main__":
    # 初始化避障系统
    avoidance_system = DroneObstacleAvoidance(
        order_file='nanjing_delivery_700_2026_01.csv',
        building_file='nanjing_buildings.csv'
    )
    
    # 处理第一个订单
    path, obstacles = avoidance_system.process_first_order()

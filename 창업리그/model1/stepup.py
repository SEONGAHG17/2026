import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 설정 파일 경로 지정
    pkg_dir = get_package_share_directory('concert_model1')
    octomap_params_path = os.path.join(pkg_dir, 'config', 'octomap_params.yaml')

    return LaunchDescription([
        # 1. RTAB-Map SLAM 노드 (드론의 위치와 3D 점군 생성)
        Node(
            package='rtabmap_ros',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'frame_id': 'base_link',
                'subscribe_depth': False,
                'subscribe_laser_scan': True,
                'approx_sync': True,
                'use_sim_time': True
            }],
            remappings=[
                ('scan', '/lidar/scan'),
                ('odom', '/rtabmap/odom')
            ]
        ),

        # 2. OctoMap 서버 노드 (점군 데이터를 받아 .bt 격자 지도로 변환)
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[octomap_params_path],
            remappings=[
                ('cloud_in', '/rtabmap/cloud_map') # RTAB-Map이 만든 맵을 OctoMap 인풋으로 연결
            ]
        )
    ])
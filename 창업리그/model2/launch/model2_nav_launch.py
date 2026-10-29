import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_dir = get_package_share_directory('concert_model2')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # 설정 파일 및 지도 경로 지정
    nav2_params_path = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    map_file_path = os.path.expanduser('~/concert_model1/maps/concert_stage.bt')

    return LaunchDescription([
        # 1. OctoMap을 이용해 .bt 지도를 Nav2에 공급하는 서버 노드
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[{
                'resolution': 0.05,
                'frame_id': 'map',
                'sensor_model/max_range': 10.0
            }],
            arguments=[map_file_path]
        ),

        # 2. Nav2 네비게이션 스택 통합 런치 호출
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'true',
                'params_file': nav2_params_path
            }.items()
        )
    ])
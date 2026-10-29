import os
from glob import glob
from setuptools import setup

package_name = 'concert_model2'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # 패키지 리소스 등록
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 런치 파일 폴더 내의 모든 .py 파일을 빌드 시 포함
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        
        # 설정 파일 폴더 내의 모든 .yaml 파일을 빌드 시 포함
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Seong-ah Gwak',
    maintainer_email='seongah@domain.com',
    description='Concert drone navigation and control pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 파이썬 커스텀 노드가 추가될 경우 여기에 실행 경로를 등록합니다.
        ],
    },
)
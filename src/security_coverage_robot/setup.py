import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'security_coverage_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models', 'urdf'), glob('models/urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Security coverage robot with color-zone awareness',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = security_coverage_robot.perception_node:main',
            'planner_node = security_coverage_robot.planner_node:main',
            'control_node = security_coverage_robot.control_node:main',
            'metrics_node = security_coverage_robot.metrics_node:main',
        ],
    },
)

# test push
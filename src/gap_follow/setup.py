from setuptools import find_packages, setup

package_name = 'gap_follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Punit Raveendran',
    maintainer_email='punitraveendran@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = gap_follow.pure_pursuit_node:main',
            'stanley_node = gap_follow.stanley_node:main',
            'lap_timer_node = gap_follow.lap_timer_node:main',
            'opponent_driver_node = gap_follow.opponent_driver_node:main',
            'opponent_detector_node = gap_follow.opponent_detector_node:main',
            'spliner_node = gap_follow.spliner_node:main',
            'state_machine_node = gap_follow.state_machine_node:main',
        ],
    },
)

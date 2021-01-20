from setuptools import setup

description = """MapleWrapper allows real-time game data extraction for MapleStory v.92 and below 
clients. It is primarily intended to facilitate the production of reinforcement learning environments
for game agents. 

For more information, read the README at https://github.com/vinmorel/MapleWrapper.
"""

setup(
    name='MapleWrapper',
    version='1.0.0',
    description='MapleStory v.92 and below client wrapper',
    long_description=description,
    author='Vincent Morel',
    license='MIT License',
    keywords='MapleStory Wrapper Reinforcement Learning',
    url='https://github.com/vinmorel/MapleWrapper',
    packages=[
        'maplewrapper', 
        'maplewrapper.utils', 
    ],
    install_requires=[
        'd3dshot',
        'numpy>=1.19.1',
        'opencv-python>=4.4.0.42',
        'pywin32==225',
        'pywin32-ctypes>=0.2.0',
        'requests>=2.23.0'
    ],
    package_data={'maplewrapper': ['templates/general/*png', 
                                    'templates/nametag_characters/*png',
                                    'templates/numbers/*png',
                                    'templates/numbers_lvl/*png',
                                    'templates/numbers_hitreg/*png',
                                    'utils/*txt']},
    
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: MIT License'
    ],
)
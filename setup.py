from setuptools import setup, find_packages

setup(
    name='fastllm',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy',
        'pandas',
        # Add more dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            # If you have any command-line scripts, list them here
            # 'script_name=module:function',
        ],
    },
)

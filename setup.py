from setuptools import setup, find_packages


def read_requirements(file_path):
    with open(file_path, "r") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="fastllm",
    version="1.0",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            # If you have any command-line scripts, list them here
            # 'script_name=module:function',
        ],
    },
)

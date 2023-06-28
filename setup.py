from setuptools import setup, find_packages

setup(
    name="LeGM",
    version="1.0.2",
    description="Your general ML experiment manager",
    author="Georgios Chochlakis",
    author_email="chochlak@usc.edu",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tensorboard",
    ],
    extras_require={"dev": ["black", "pytest"]},
    entry_points={
        "console_scripts": [
            "best_from_logs = legm.logging_utils:get_best",
        ]
    },
)

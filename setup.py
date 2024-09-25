from setuptools import setup, find_packages

setup(
    name="LeGM",
    version="1.4.18",
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
        "gridparse @ git+https://github.com/gchochla/gridparse.git@main",
    ],
    extras_require={"dev": ["black", "pytest"]},
    entry_points={
        "console_scripts": [
            "best_from_logs = legm.exp_manager:get_best",
        ]
    },
)

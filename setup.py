from setuptools import setup, find_packages

setup(
    name="ds_lab_app",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "pytest",
        "flake8",
        "ruff",
        "pytest-cov"
    ],
)

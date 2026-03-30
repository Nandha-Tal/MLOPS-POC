from setuptools import find_packages, setup

setup(
    name="mlops_poc",
    version="1.0.0",
    description="K8s Cluster Monitoring — MLOps + Agentic AI POC",
    author="MLOps Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "mlflow>=2.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.9",
        "jinja2>=3.1.0",
        "pyyaml>=6.0",
        "python-box>=7.0.0",
        "ensure>=1.0.4",
        "python-dotenv>=1.0.0",
        "anthropic>=0.25.0",
        "requests>=2.31.0",
        "httpx>=0.27.0",
    ],
)

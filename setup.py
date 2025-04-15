# setup.py

import setuptools
import os

# --- 项目元数据 ---
NAME = "ptforge"
VERSION = "0.1.2"
AUTHOR = "AlbertCui / pforge.ai"
EMAIL = "albertzhouzhou@gmail.com"
DESCRIPTION = "A framework for automated LLM prompt optimization."
URL = "https://github.com/pforge-ai/prompt-forge"
REQUIRES_PYTHON = ">=3.8"

# --- 依赖项 ---
# 项目运行所必需的基础依赖 (Core dependencies required for the library to run)
REQUIRED = [
    "httpx>=0.20.0,<1.0.0", # For OpenAIClient (and potentially others)
    # 添加其他核心依赖 (Add other core dependencies here)
]

# --- 读取 README 作为长描述 ---
# --- Read README for long description ---
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION # Fallback to short description if README is missing

# --- setuptools 配置 ---
# --- setuptools configuration ---
setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(exclude=["tests*", "examples*"]),
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 3 - Alpha",
    ],
    keywords="llm prompt engineering optimization automatic tuning ai",
)
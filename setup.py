# coding=utf-8
import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except:
    long_description = '''An extension for geppy to learn and resample a start population from different cases, forked from https://github.com/ShuhuaGao/geppy'''

setuptools.setup(
    name="geppy-memory",
    version="0.0.1",
    license='LGPL-3.0 License',
    author="Max Reissmann",
    author_email="reissmannm@student.unimelb.edu.au",
    description=".",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
	long_description_content_type='text/markdown',
    keywords=['evolutionary computation', 'gene expression programming',
              'computational intelligence', 'genetic programming','llm initialisation'],
    packages=setuptools.find_packages(),
    install_requires=['deap','numpy','pandas','joblib','matplotlib','ucimlrepo','numexpr','sympy','tf-models-official==2.7.0', 'tensorflow==2.11.0','rotary-embedding-tensorflow',"uci-dataset",'ucimlrepo']
)


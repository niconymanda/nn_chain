from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "knn_chain.pyx", compiler_directives={"language_level": "3"}
    )
)
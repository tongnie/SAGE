from setuptools import Extension, setup

try:
    import numpy as np
    from Cython.Build import cythonize
except ImportError as exc:
    raise RuntimeError("Install build dependencies with `pip install -e .` from the repository root.") from exc


extensions = [
    Extension(
        "advgen.utils_cython",
        ["advgen/utils_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]


setup(ext_modules=cythonize(extensions, language_level=3))

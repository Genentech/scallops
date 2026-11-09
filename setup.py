import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "scallops.segmentation._propagate",
        sources=["scallops/segmentation/_propagate.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++",
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3", "embedsignature": True},
    )
)

import os

import numpy
from setuptools import Extension, setup

# For PEP 517 builds with pyproject.toml, Cython specified in
# [build-system].requires should be available when this script runs.
try:
    from Cython.Build import cythonize

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    # This fallback means we can only build from pre-generated .c or .cpp files.
    # For development and CI, Cython should always be available.
    print(
        "Warning: Cython not found. "
        "Will attempt to build from existing .cpp files if available."
    )

numpy_include_dir = numpy.get_include()


def get_source_file(base_path_and_module, ext_lang="c++"):
    """
    Determines whether to use the .pyx file (if Cython is available)
    or the pre-compiled .cpp/.c file.
    """
    pyx_file = os.path.join(*base_path_and_module.split(".")) + ".pyx"
    # Corresponding C/C++ file (Cython converts .pyx to .cpp if language="c++")
    compiled_ext = ".cpp" if ext_lang == "c++" else ".c"
    compiled_file = os.path.join(*base_path_and_module.split(".")) + compiled_ext

    if HAS_CYTHON and os.path.exists(pyx_file):
        return [pyx_file]
    elif os.path.exists(compiled_file):
        print(f"Using pre-existing {compiled_file} (Cython not run or .pyx not found)")
        return [compiled_file]
    elif os.path.exists(pyx_file):
        # .pyx exists but Cython is not available (or not used for some reason)
        raise RuntimeError(
            f"Found {pyx_file} but Cython is not available/enabled to compile it, "
            f"and {compiled_file} does not exist."
        )
    else:
        raise FileNotFoundError(
            f"Neither {pyx_file} nor {compiled_file} found for module {base_path_and_module}."
        )


extensions = [
    Extension(
        "scallops.segmentation._propagate",
        sources=get_source_file("scallops.segmentation._propagate", "c++"),
        include_dirs=[numpy_include_dir] if numpy_include_dir else [],
        language="c++",
    )
]

if HAS_CYTHON:
    # Filter extensions to only those that are actual .pyx files for Cythonizing
    pyx_extensions = [
        ext for ext in extensions if any(s.endswith(".pyx") for s in ext.sources)
    ]
    if pyx_extensions:
        print("Cythonizing .pyx extension modules...")
        extensions = cythonize(
            pyx_extensions,
            compiler_directives={
                "language_level": "3",  # From your pyproject.toml (requires-python = ">=3.11")
                "embedsignature": True,  # Useful for docstrings
            },
            # force=True, # Uncomment to always re-cythonize, useful for debugging CI
        )
    else:
        print("No .pyx files found in extensions to Cythonize directly by this script.")

setup(
    # name="scallops", # Usually picked up from pyproject.toml
    ext_modules=extensions,
    # setup_requires is mostly for legacy direct `setup.py` calls without pip intervention.
    # For PEP517 builds, [build-system].requires in pyproject.toml is used.
    # If you want to ensure Cython/NumPy are available for direct setup.py calls:
    # setup_requires=[
    #    'setuptools>=42',
    #    'cython>=0.29', # Or match your pyproject.toml's [build-system]
    #    'numpy<2.0',    # Or match your pyproject.toml's [build-system] and runtime deps
    # ],
    zip_safe=False,  # Good practice for packages with C extensions
)

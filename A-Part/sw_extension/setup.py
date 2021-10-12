from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="swextension",
    version='0.0.1',
    ext_modules=[
        CppExtension(
            "swextension",
            include_dirs=["./swops/"],
            library_dirs=["./swops/"],
            sources=["swextension.cpp"],
            extra_link_args=["-lswops", "-lm", "-lm_slave"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

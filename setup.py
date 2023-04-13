import numpy as np  # 导入 NumPy 库  
from setuptools import setup, Extension  # 导入 setuptools 库和 Extension 类  
from Cython.Build import cythonize  # 导入 Cython.Build 库并定义 cythonize 函数

setup(  
    ext_modules=cythonize(  # 调用 cythonize 函数将所有的 pyx 文件转换为可执行的 C++代码  
        Extension("*",  # 指定所有的 pyx 文件作为 Extension 模块  
            ["reckit/cython/*.pyx"],  # 指定所有的 pyx 文件所在的目录  
            language="c++",  # 指定使用 C++语言编写模块  
            extra_compile_args=["-std=c++11"],  # 指定编译参数  
        ),  
        compiler_directives={  # 定义编译指令  
            "language_level": 3,  # 指定语言级别为 3  
            "boundscheck": False,  # 禁用数组边界检查  
            "wraparound": False,  # 禁用数组 wraparound 检查  
        },  
    ),  
    include_dirs=[np.get_include()],  # 指定 include 文件夹的路径  
)

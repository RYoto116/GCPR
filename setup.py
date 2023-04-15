
# import os
# import shutil
# import numpy as np
# from functools import wraps
# from Cython.Build import cythonize
# from distutils.core import setup, Extension


# def get_include_dirs(workspace):
#     head_dirs = [np.get_include()]
#     for root, dirs, files in os.walk(workspace):
#         for file in files:
#             if file.endswith("h") or file.endswith("hpp"):
#                 head_dirs.append(root)

#     return list(set(head_dirs))


# def get_extensions(workspace):
#     exts = []

#     for root, dirs, files in os.walk(workspace):
#         for file in files:
#             if file.endswith("pyx"):
#                 pyx_file = os.path.join(root, file)
#                 pyx_path = pyx_file[:-4].split(os.sep)
#                 pyx_path = pyx_path[1:] if pyx_path[0] == '.' else pyx_path[1:]
#                 name = ".".join(pyx_path)

#                 ext = Extension(name, [pyx_file],
#                                 extra_compile_args=["-std=c++11"])
#                 exts.append(ext)
#     return exts


# def clean(func):
#     """clean intermediate file
#     """

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         def _clean_file():
#             new_dirs = set()
#             new_files = set()
#             for root, dirs, files in os.walk("."):
#                 new_dirs.update([os.path.join(root, d) for d in dirs])
#                 new_files.update([os.path.join(root, f) for f in files])

#             for new_dir in new_dirs:
#                 if os.path.exists(new_dir) and new_dir not in old_dirs \
#                         and "dist" not in new_dir.split(os.path.sep):
#                     shutil.rmtree(new_dir)
#             for new_file in new_files:
#                 if os.path.exists(new_file) and new_file not in old_files \
#                         and not new_file.endswith("so") and not new_file.endswith("pyd"):
#                     os.remove(new_file)

#         old_dirs = set()
#         old_files = set()
#         for root, dirs, files in os.walk("."):
#             old_dirs.update([os.path.join(root, d) for d in dirs])
#             old_files.update([os.path.join(root, f) for f in files])

#         try:
#             result = func(*args, **kwargs)
#         except Exception as e:
#             _clean_file()
#             raise e

#         _clean_file()
#         return result

#     return wrapper


# # @clean
# def compile_cpp():
#     extensions = get_extensions(".")
#     include_dirs = get_include_dirs(".")
#     module_list = cythonize(extensions, language="c++", annotate=False, compiler_directives={  # 定义编译指令  
#             "language_level": 3,  # 指定语言级别为 3  
#             "boundscheck": False,  # 禁用数组边界检查  
#             "wraparound": False,  # 禁用数组 wraparound 检查  
#         },
# )

#     setup(ext_modules=module_list, include_dirs=include_dirs)

# if __name__ == "__main__":
#     compile_cpp()

import numpy as np  # 导入 NumPy 库  
from setuptools import setup, Extension  # 导入 setuptools 库和 Extension 类  
from Cython.Build import cythonize  # 导入 Cython.Build 库并定义 cythonize 函数

setup(  
    ext_modules=cythonize(  # 调用 cythonize 函数将所有的 pyx 文件转换为可执行的 C++代码  
        Extension(
            "*",  # 指定所有的 pyx 文件作为 Extension 模块  
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
    include_dirs=["reckit/cython/include/", np.get_include()],  # 指定 include 文件夹的路径  
)
import os
import platform
import shutil
import sys
import glob
from distutils.command.build import build
from distutils.command.install import install
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# ===== Stage shared libraries into source pybicos directory for packaging =====
project_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(project_dir, 'build')
if not os.path.exists(build_dir):
    build_dir = os.path.join(project_dir, 'build')
if os.path.exists(build_dir):
    print("Build directory:", build_dir)
    print("Project directory:", project_dir)
    stage_dir = os.path.join(project_dir, 'pybicos')
    os.makedirs(stage_dir, exist_ok=True)
    for prefix in ['pybicos_c', 'libBICOS']:
        # Linux .so and versioned
        for f in glob.glob(os.path.join(build_dir, prefix + '.so*')):
            print("so:", f)
            shutil.copy(f, stage_dir)
        # Windows DLL
        for f in glob.glob(os.path.join(build_dir, prefix + '*.dll')):
            print("dll:", f)
            shutil.copy(f, stage_dir)
        # macOS dylib
        for f in glob.glob(os.path.join(build_dir, prefix + '*.dylib')):
            shutil.copy(f, stage_dir)
# ===== End staging =====

class CustomBuild(build):
    def run(self):
        # Run the original build command
        build.run(self)

        # Copy the shared library to the build directory
        self.copy_shared_library()
        
        # ===== Manually stage shared libraries for packaging =====
        project_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(project_dir, 'build', 'cmake')
        if not os.path.exists(build_dir):
            build_dir = os.path.join(project_dir, 'build')
        if os.path.exists(build_dir):
            for prefix in ['pybicos_c', 'libBICOS']:
                for fpath in glob.glob(os.path.join(build_dir, prefix + '*.so*')):
                    shutil.copy(fpath, os.path.join(project_dir, 'pybicos'))
                for fpath in glob.glob(os.path.join(build_dir, prefix + '*.dll')):
                    shutil.copy(fpath, os.path.join(project_dir, 'pybicos'))
                for fpath in glob.glob(os.path.join(build_dir, prefix + '*.dylib')):
                    shutil.copy(fpath, os.path.join(project_dir, 'pybicos'))
        # ===== End staging =====
    
    
    
    def copy_shared_library(self):
        # Determine source and destination paths
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'cmake')
        package_dir = os.path.join(self.build_lib, 'pybicos')
        os.makedirs(package_dir, exist_ok=True)
        
        # Copy __init__.py so package is importable from build directory
        src_init = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pybicos_temp', '__init__.py')
        dst_init = os.path.join(package_dir, '__init__.py')
        shutil.copy(src_init, dst_init)
        
        # Copy the shared library
        if platform.system() == 'Windows':
            lib_name = 'pybicos_c.dll'
            lib_path = os.path.join(build_dir, 'Release', lib_name)
            # Also copy BICOS.dll
            bicos_lib = 'BICOS.dll'
            bicos_path = os.path.join(build_dir, 'Release', bicos_lib)
            if os.path.exists(bicos_path):
                shutil.copy(bicos_path, package_dir)
            # Copy OpenCV DLLs if they exist
            opencv_dlls = ['opencv_world4*.dll', 'opencv_core4*.dll', 'opencv_imgproc4*.dll']
            for dll in opencv_dlls:
                try:
                    import glob
                    opencv_path = os.environ.get('OPENCV_DIR', '')
                    if opencv_path:
                        opencv_bin = os.path.join(opencv_path, 'bin')
                        for f in glob.glob(os.path.join(opencv_bin, dll)):
                            shutil.copy(f, package_dir)
                except Exception as e:
                    print(f"Warning: Could not copy OpenCV DLL: {e}")
        elif platform.system() == 'Darwin':
            lib_name = 'libpybicos_c.dylib'
            lib_path = os.path.join(build_dir, lib_name)
        else:
            lib_name = 'pybicos_c.so'
            lib_path = os.path.join(build_dir, lib_name)
        
        if os.path.exists(lib_path):
            shutil.copy(lib_path, package_dir)
        else:
            print(f"Warning: Could not find shared library at {lib_path}")


class CustomInstall(install):
    def run(self):
        # First invoke build (CMake + shared libs) before install
        self.run_command('build')
        install.run(self)
        # Determine build directory (CMake build)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(project_dir, 'build')
        if not os.path.exists(build_dir):
            build_dir = os.path.join(project_dir, 'build')
        # Copy shared libraries into the installed pybicos package
        # Destination package directory
        pkg_dir = os.path.join(self.install_lib, 'pybicos')
        print("Package directory:", pkg_dir)
        os.makedirs(pkg_dir, exist_ok=True)
        print("Build directory:", build_dir)
        # Copy all generated shared libraries into the package
        for prefix in ['pybicos_c', 'libBICOS']:
            # Linux .so and versioned .so.X
            for fpath in glob.glob(os.path.join(build_dir, prefix + '*.so*')):
                shutil.copy(fpath, pkg_dir)
            # Windows .dll
            for fpath in glob.glob(os.path.join(build_dir, prefix + '*.dll')):
                shutil.copy(fpath, pkg_dir)
            # macOS .dylib
            for fpath in glob.glob(os.path.join(build_dir, prefix + '*.dylib')):
                shutil.copy(fpath, pkg_dir)
        

setup(
    name="pybicos",
    version="2.2.0",
    description="Python bindings for libBICOS using ctypes",
    author="Robotics Group @ Julius-Maximilian University",
    packages=["pybicos"],
    include_package_data=True,
    package_data={"pybicos": ["*.so", "*.dll", "*.dylib"]},
    cmdclass={
        'build': CustomBuild,
        'install': CustomInstall,
    },
)

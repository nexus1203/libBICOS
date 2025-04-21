# libBICOS

BInary COrrespondence Search for multi-shot stereo imaging, with optional CUDA acceleration.
Fork modified.. cog, and fmt dependencies are removed. some parts are modified in cpu.cpp to avoid stack overflow.
## Citing:

This is the implementation of the [corresponding paper](https://isprs-archives.copernicus.org/articles/XLVIII-2-W7-2024/57/2024/isprs-archives-XLVIII-2-W7-2024-57-2024.pdf) which appeared in [Optical 3D Metrology 2024](https://o3dm.fbk.eu):
```bibtex
@article{liebender2024libbicos,
  title={libBICOS -- An Open Source GPU-Accelerated Library implementing BInary COrrespondence Search for 3D Reconstruction},
  author={Liebender, Christoph and Bleier, Michael and N{\"u}chter, Andreas},
  journal={The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={48},
  pages={57--64},
  year={2024},
  publisher={Copernicus Publications G{\"o}ttingen, Germany}
}
```

## Build and Installation

The build system has been switched to CMake, eliminating Meson, cog, and Google Benchmark as dependencies. You now only need:
- CMake 3.10 or higher
- A C++17-compatible compiler (GCC ≥7.0, Visual Studio 2019+, or clang)
- OpenCV 4.x (CUDA-enabled for GPU builds)
- (Optional) NVIDIA CUDA Toolkit 12.3 or higher ( the kernels uses newer primitives and calls)

```bash
# Clone the latest release
git clone --depth 1 --branch v2.2.0 https://github.com/nexus1203/libBICOS
cd libBICOS

# Create and enter build directory
mkdir build && cd build

# CPU-only build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBICOS_IMPLEMENTATION=CPU
# For CUDA acceleration
cmake .. -DCMAKE_BUILD_TYPE=Release -DBICOS_IMPLEMENTATION=CUDA

# Compile
make -j$(nproc)

# (Optional) Install to system
sudo make install

# Install Python bindings
cd ..
pip install .      # regular install
pip install -e .   # editable install
```

The versioning scheme of the library is [Semantic Versioning](https://semver.org/).

## Usage

### Linking
After installing, you can include `/usr/local/include/BICOS/*` and link against `/usr/local/lib/libBICOS.so`.

### Python module
With an available python installation, CMake will build a working ctype library  `pybicos` that you can install with pip. It is a ctypes wrapper around the C++ API for more convenient experimentation:
```python
import pybicos
import cv2 as cv

lstack = [cv.imread(f"data/left/{i}.png", cv.IMREAD_UNCHANGED) for i in range(20)]
rstack = [cv.imread(f"data/right/{i}.png", cv.IMREAD_UNCHANGED) for i in range(20)]

cfg = pybicos.Config()
cfg.nxcorr_threshold = 0.9

disparity, correlation_map = pybicos.match(lstack, rstack, cfg)

```

### Commandline-interface
Alternatively, this project builds `bicos-cli`. To give you a feel for the parameters of BICOS, you can download an example dataset with [`data/prepare.sh`](/data/prepare.sh) that contains rectified imagery from a stereo camera, where images per side only differ in the projected light pattern.
Calling:
```console
$ bicos-cli data/{left,right} -q data/Q.yaml --threshold 0.96 --stacksize 33 --limited --variance 2.0 --step 0.1 -o /tmp/result.png
```
will get you:

#### Disparity
![Example disparity](/example-disp.png)

#### Pointcloud
![Example pointcloud](/example-pcl.png)

While only requiring ~44ms (RTX4090) and ~1.6GB VRAM for matching on two stacks of 33 images each.

The most significant parameters can be summarized:

- `--threshold`: increasing this reduces match-outliers. High values rely on a reliable pattern.
- `--variance`: high values reduce coarse outliers, e.g. pixels where no pattern was projected. May reduce correct matches on low pattern contrast.
- `--step`: optional value for subpixel-interpolation around matches.
- `--lr-maxdiff`: use a maximum left-right disparity difference as a postfilter instead of no-duplicates.

Other settings are available; Try `bicos-cli -h` for details.

## Benchmarking

After building the project:
```bash
# From the build directory:
ctest -C Release --verbose
```

## Light projection:
For starters, you may find https://github.com/Sir-Photch/VRRTest useful for projecting a light pattern using an aftermarket projector.

## License

This library is licensed under the GNU Lesser General Public License v3.0 (or later).
Please see [COPYING](/COPYING) and [COPYING.LESSER](/COPYING.LESSER) for details.

![LGPL-3.0-Logo](https://www.gnu.org/graphics/lgplv3-147x51.png)

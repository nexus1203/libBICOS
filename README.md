# libBICOS

BInary COrrespondence Search for multi-shot stereo imaging, with optional CUDA acceleration.

## Citing:

This is the implementation of the corresponding paper to appear in [Optical 3D Metrology](https://o3dm.fbk.eu). *BibTeX will appear here when available.*

## Build:

Dependencies:

- `gcc` or equivalent C compiler with C++20 (C++17 experimental) support (build)
- [`meson`](https://github.com/mesonbuild/meson) >= 1.1.0 (build)
- `opencv 4.x` with cuda support
- `cuda 12.x` including toolkit 
- [`fmt`](https://github.com/fmtlib/fmt) (C++17)
- [`cxxopts`](https://github.com/jarro2783/cxxopts) (optional, for cli)
- [`benchmark`](https://github.com/google/benchmark) (optional, for executing benchmarks)

```bash
# recommended: clone the most recent release
$ git clone --depth 1 --branch v1.2.1 https://github.com/JMUWRobotics/libBICOS
$ cd libBICOS

$ meson setup builddir --buildtype release

# optional but recommended if you have access to a CUDA-capable GPU
$ meson configure -Dimplementation=cuda builddir

$ meson install -C builddir
```

The versioning scheme of the library is [Semantic Versioning](https://semver.org/).

## Usage

### Linking
After installing, you can include `/usr/local/include/BICOS/*` and link against `/usr/local/lib/libBICOS.so`.

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

While only requiring ~110ms (RTX4090) and ~1.6GB VRAM for matching on two stacks of 33 images each.

The most significant parameters can be summarized:

- `--threshold`: increasing this reduces match-outliers. High values rely on a reliable pattern.
- `--variance`: high values reduce coarse outliers, e.g. pixels where no pattern was projected. May reduce correct matches on low pattern contrast.
- `--step`: optional value for subpixel-interpolation around matches.

Other settings are available; Try `bicos-cli -h` for details.

## Benchmarking:

```console
$ meson test --benchmark -C builddir --interactive
```

## Light projection:
For starters, you may find https://github.com/Sir-Photch/VRRTest useful for projecting a light pattern using an aftermarket projector.

## License

This library is licensed under the GNU Lesser General Public License v3.0 (or later).
Please see [COPYING](/COPYING) and [COPYING.LESSER](/COPYING.LESSER) for details.

![LGPL-3.0-Logo](https://www.gnu.org/graphics/lgplv3-147x51.png)

# libBICOS

BInary COrrespondence Search for multi-shot stereo imaging, with optional CUDA acceleration.

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

## Build:

Dependencies:

- `gcc` or equivalent C compiler with C++17 support (build)
- [`meson`](https://github.com/mesonbuild/meson) >= 1.1.0 (build)
- `opencv 4.x` with cuda support
- `cuda 12.x` including toolkit
- [`benchmark`](https://github.com/google/benchmark) (optional, for executing benchmarks)

```bash
# recommended: clone the most recent release
$ git clone --depth 1 --branch v2.0.0 https://github.com/JMUWRobotics/libBICOS
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

While only requiring ~44ms (RTX4090) and ~1.6GB VRAM for matching on two stacks of 33 images each.

The most significant parameters can be summarized:

- `--threshold`: increasing this reduces match-outliers. High values rely on a reliable pattern.
- `--variance`: high values reduce coarse outliers, e.g. pixels where no pattern was projected. May reduce correct matches on low pattern contrast.
- `--step`: optional value for subpixel-interpolation around matches.
- `--lr-maxdiff`: use a maximum left-right disparity difference as a postfilter instead of no-duplicates.

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

#!/usr/bin/env bash
##  libBICOS: binary correspondence search on multishot stereo imagery
##  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU Lesser General Public License as
##  published by the Free Software Foundation, either version 3 of the
##  License, or (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU Lesser General Public License for more details.
##
##  You should have received a copy of the GNU Lesser General Public License
##  along with this program.  If not, see <https://www.gnu.org/licenses/>.
##

set -xeu

buildtype=$(meson introspect builddir --buildoptions | grep -Po '(?<=\{"name": "buildtype", "value": ")[a-z]+(?=")')

builddir/bicos-cli data/left data/right -t 0.9 -s 0.1 -n 12 --double -o /tmp/bicos-regress
diff "test/regress_cuda_${buildtype:-release}.tiff" /tmp/bicos-regress.tiff

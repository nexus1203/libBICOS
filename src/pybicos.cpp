/**
 *  libBICOS: binary correspondence search on multishot stereo imagery
 *  Copyright (C) 2024-2025  Robotics Group @ Julius-Maximilian University
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "common.hpp"
#include "match.hpp"
#include "opencv2/core/cuda.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace BICOS;

static Image handle_to_image(const py::handle& hdl) {
    auto arr = hdl.cast<py::array>();
    int dtype = arr.dtype().num();

    int type;
    if (dtype == py::dtype::of<uint8_t>().num())
        type = CV_8UC1;
    else if (dtype == py::dtype::of<uint16_t>().num())
        type = CV_16UC1;
    else
        throw std::invalid_argument("dtype must be uint{8,16}");

    const cv::Mat header(
        (int)arr.shape(0),
        (int)arr.shape(1),
        type,
        const_cast<void*>(arr.data()),
        (int)arr.strides(0)
    );
#ifdef BICOS_CUDA
    return Image(header);
#else
    return header;
#endif
}

static py::array image_to_buffer(const Image& img) {
    py::array ret;
    switch (img.type()) {
    case CV_16SC1:
        ret = py::array_t<int16_t>({ img.rows, img.cols });
        break;
    case CV_32FC1:
        ret = py::array_t<float>({ img.rows, img.cols });
        break;
    default:
        __builtin_unreachable();
    }
    cv::Mat header(img.rows, img.cols, img.type(), const_cast<void*>(ret.data()), ret.strides(0));
#ifdef BICOS_CUDA
    img.download(header);
#else
    img.copyTo(header);
#endif
    return ret;
}

py::array pybicos_match(py::list stack0, py::list stack1, Config cfg) {
    std::vector<Image> stack0_, stack1_;
    Image disparity;

    if (stack0.empty() || stack1.empty())
        throw Exception("empty stacks");

    std::transform(stack0.begin(), stack0.end(), std::back_inserter(stack0_), handle_to_image);
    std::transform(stack1.begin(), stack1.end(), std::back_inserter(stack1_), handle_to_image);

    BICOS::match(stack0_, stack1_, disparity, cfg);

    return image_to_buffer(disparity);
}

PYBIND11_MODULE(pybicos, m) {
    m.doc() = "Python bindings for libBICOS";

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("nxcorr_threshold", &Config::nxcorr_threshold)
        .def_readwrite("subpixel_step", &Config::subpixel_step)
        .def_readwrite("min_variance", &Config::min_variance)
        .def_readwrite("mode", &Config::mode)
#ifdef BICOS_CUDA
        .def_readwrite("precision", &Config::precision)
#endif
        .def_readwrite("variant", &Config::variant);

#ifdef BICOS_CUDA
    py::enum_<Precision>(m, "Precision").def(py::init<>());
#endif

    py::class_<Variant::NoDuplicates>(m, "Variant_NoDuplicates").def(py::init<>());
    py::class_<Variant::Consistency>(m, "Variant_Consistency").def(py::init<int, bool>());

    m.def("match", &pybicos_match);
}
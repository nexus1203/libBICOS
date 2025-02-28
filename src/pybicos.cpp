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
#include "formatable.hpp"
#include "match.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define ExceptionFmt(fmtstr, ...) Exception(fmt::format(fmtstr, ##__VA_ARGS__))

namespace py = pybind11;
using namespace BICOS;

template<>
struct fmt::formatter<py::str>: formatter<string_view> {
    auto format(const py::str& str, format_context& ctx) const -> format_context::iterator {
        const std::string& s = str;
        return format_to(ctx.out(), "{}", s);
    }
};

static int cvtype_of(py::dtype pydtype) {
    int dtype = pydtype.num();

    if (dtype == py::dtype::of<uint8_t>().num())
        return CV_8U;
    else if (dtype == py::dtype::of<uint16_t>().num())
        return CV_16U;
    else if (dtype == py::dtype::of<int16_t>().num())
        return CV_16S;
    else if (dtype == py::dtype::of<float>().num())
        return CV_32F;
    else if (dtype == py::dtype::of<double>().num())
        return CV_64F;
    else
        throw ExceptionFmt("unimplemented cvtype_of(dtype={})", py::str(pydtype));
}

static py::dtype dtype_of(int cvtype) {
    switch (cvtype) {
        case CV_16SC1:
            return py::dtype::of<int16_t>();
        case CV_32FC1:
            return py::dtype::of<float>();
        case CV_64FC1:
            return py::dtype::of<double>();
        default:
            throw ExceptionFmt("unimplemented dtype_of(cvtype={})", cvtype);
    }
}

static Image handle_to_image(const py::handle& hdl) {
    auto arr = hdl.cast<py::array>();

    const cv::Mat header(
        (int)arr.shape(0),
        (int)arr.shape(1),
        cvtype_of(arr.dtype()),
        const_cast<void*>(arr.data()),
        (int)arr.strides(0)
    );

#ifdef BICOS_CUDA
    return Image(header);
#else
    return header;
#endif
}

static void image_to_array(const Image& img, py::array& ret) {
    int type = img.type();
    if (ret.shape(0) != img.rows || ret.shape(1) != img.cols || cvtype_of(ret.dtype()) != type)
        ret = py::array(dtype_of(type), { img.rows, img.cols });

    cv::Mat header(img.rows, img.cols, type, ret.mutable_data(), ret.strides(0));
#ifdef BICOS_CUDA
    img.download(header);
#else
    if (img.data != header.data)
        img.copyTo(header);
#endif
}

py::tuple pybicos_match(py::list stack0, py::list stack1, Config cfg) {
    py::array disparity, corrmap;
    std::vector<Image> stack0_, stack1_;
    Image disparity_, corrmap_, *pcorrmap_ = &corrmap_;

    if (stack0.empty() || stack1.empty())
        throw Exception("empty stacks");

    std::transform(stack0.begin(), stack0.end(), std::back_inserter(stack0_), handle_to_image);
    std::transform(stack1.begin(), stack1.end(), std::back_inserter(stack1_), handle_to_image);

    BICOS::match(stack0_, stack1_, disparity_, cfg, pcorrmap_);

    image_to_array(disparity_, disparity);
    image_to_array(corrmap_, corrmap);

    return py::make_tuple(disparity, corrmap);
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
        .def_readwrite("variant", &Config::variant)
        .def("__repr__", [](const Config& c) { return fmt::format("{}", c); });

#ifdef BICOS_CUDA
    py::enum_<Precision>(m, "Precision")
        .value("Double", Precision::DOUBLE)
        .value("Single", Precision::SINGLE)
        .export_values();
#endif

    auto Variant = m.def_submodule("Variant");
    {
        py::class_<Variant::NoDuplicates>(Variant, "NoDuplicates")
            .def(py::init())
            .def("__repr__", [](const Variant::NoDuplicates& nd) { return fmt::format("{}", nd); });
        py::class_<Variant::Consistency>(Variant, "Consistency")
            .def(py::init<int, bool>(), py::arg("max_lr_diff") = 1, py::arg("no_dupes") = false)
            .def("__repr__", [](const Variant::Consistency& vc) { return fmt::format("{}", vc); });
    }

    py::enum_<TransformMode>(m, "TransformMode")
        .value("Full", TransformMode::FULL)
        .value("Limited", TransformMode::LIMITED)
        .export_values();

    py::register_local_exception<Exception>(m, "Exception");

    m.def(
        "match",
        &pybicos_match,
        py::arg("stack0").none(false),
        py::arg("stack1").none(false),
        py::arg_v("cfg", Config {}, "Default")
    );

    m.def("invalid_disparity", [](py::dtype dtype) -> py::object {
        switch (cvtype_of(dtype)) {
            case CV_16S:
                return py::int_(INVALID_DISP<int16_t>);
            case CV_32F:
                return py::float_(INVALID_DISP<float>);
            default:
                throw ExceptionFmt("invalid_disparity undefined for {}", py::str(dtype));
        }
    });
}
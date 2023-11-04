#include "includes/afinn/afinn.hpp"
#include "includes/heartbeat/heartbeat.hpp"
#include "includes/emotions/NED.hpp"
#include "includes/pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;

PYBIND11_MODULE(polarity, m) {
    m.doc() = "Polarity/AFINN module written in C++";

    py::class_<Polarity>(m, "Polarity")
        .def(py::init<>())
        .def("getEmoji", &Polarity::getEmoji)
        .def("getText", &Polarity::getText);
}

PYBIND11_MODULE(heartbeat, m) {
    m.doc() = "Heartbeat module written in C++";

    py::class_<Heartbeat>(m, "Heartbeat")
        .def(py::init<>())
        .def("runScan", &Heartbeat::runScan);
}

PYBIND11_MODULE(ned, m) {
    m.doc() = "NED module written in C++";

    py::class_<NED>(m, "NED")
        .def(py::init<>())
        .def("getEmotion", &NED::detectEmotion);
}

# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.22)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(firmware)

# ESP-DL model
set(MODEL_FILE
    "${CMAKE_CURRENT_LIST_DIR}/../models/model_20260929_222606_670663.espdl"
)

# Flash the model into the "model" partition.
esptool_py_flash_to_partition(
    flash
    "model"
    "${MODEL_FILE}"
)
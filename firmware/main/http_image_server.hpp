#pragma once

#include "esp_err.h"

/**
 * @brief Connect to WiFi (credentials in wifi_config.hpp) and start an HTTP
 *        server exposing GET /image.
 *
 * The response body is a raw RGB565 frame (width*height*2 bytes, little-endian)
 * and carries X-Image-Width, X-Image-Height and X-Image-Format headers.
 *
 * This module is always compiled; firmware.cpp decides whether to start it
 * (see the ENABLE_HTTP_SERVER switch there).
 *
 * @return ESP_OK on success, otherwise an error from the WiFi/HTTP bring-up.
 */
esp_err_t http_image_server_start();

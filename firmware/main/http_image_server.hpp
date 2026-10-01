#pragma once

#include "esp_err.h"

// Define to compile and run the WiFi image server (WiFi + HTTP "/image").
// Comment this out to build a detection-only firmware: no WiFi, no HTTP server.
#define ENABLE_HTTP_IMAGE_SERVER

#ifdef ENABLE_HTTP_IMAGE_SERVER
/**
 * @brief Connect to WiFi (credentials in wifi_config.hpp) and start an HTTP
 *        server exposing GET /image.
 *
 * The response body is a raw RGB565 frame (width*height*2 bytes, little-endian)
 * and carries X-Image-Width, X-Image-Height and X-Image-Format headers.
 *
 * @return ESP_OK on success, otherwise an error from the WiFi/HTTP bring-up.
 */
esp_err_t http_image_server_start();
#endif

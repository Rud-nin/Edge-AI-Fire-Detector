#include "http_image_server.hpp"

#include <cstdio>
#include <cstring>

#include "esp_log.h"
#include "esp_camera.h"

#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_http_server.h"

#include "wifi_config.hpp"

static const char *TAG = "HTTP_IMG";
static constexpr uint16_t HTTP_PORT = 80;

// ---------------------------------------------------------------------------
// GET /image -> raw RGB565 body + geometry/format headers.
// ---------------------------------------------------------------------------
static esp_err_t image_get_handler(httpd_req_t *req)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "camera capture failed");
        return ESP_FAIL;
    }

    // NOTE: httpd_resp_set_hdr() stores the pointers, it does NOT copy the
    // strings. Each header therefore needs its own buffer that stays valid
    // until the response is sent (these stack buffers do; we return after send).
    char width_value[16];
    char height_value[16];

    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");

    std::snprintf(width_value, sizeof(width_value), "%u", static_cast<unsigned>(fb->width));
    httpd_resp_set_hdr(req, "X-Image-Width", width_value);

    std::snprintf(height_value, sizeof(height_value), "%u", static_cast<unsigned>(fb->height));
    httpd_resp_set_hdr(req, "X-Image-Height", height_value);

    httpd_resp_set_hdr(req, "X-Image-Format", "rgb565");

    esp_err_t res = httpd_resp_send(req, reinterpret_cast<const char *>(fb->buf),
                                    static_cast<ssize_t>(fb->len));
    esp_camera_fb_return(fb);
    return res;
}

static httpd_handle_t start_httpd()
{
    static const httpd_uri_t image_uri = {
        .uri = "/image",
        .method = HTTP_GET,
        .handler = image_get_handler,
        .user_ctx = nullptr,
    };

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = HTTP_PORT;
    config.stack_size = 8192;
    config.lru_purge_enable = true;
    // A full frame is ~150 KB; allow slow links to finish sending.
    config.send_wait_timeout = 20;
    config.recv_wait_timeout = 20;

    httpd_handle_t server = nullptr;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "HTTP server start failed");
        return nullptr;
    }
    httpd_register_uri_handler(server, &image_uri);
    ESP_LOGI(TAG, "HTTP server listening on port %d", HTTP_PORT);
    return server;
}

// ---------------------------------------------------------------------------
// WiFi station
// ---------------------------------------------------------------------------
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disconnected, reconnecting...");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        auto *event = static_cast<ip_event_got_ip_t *>(event_data);
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

static esp_err_t wifi_init_sta()
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wifi_init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr, nullptr));

    wifi_config_t wifi_config = {};
    std::strncpy(reinterpret_cast<char *>(wifi_config.sta.ssid), WIFI_SSID,
                 sizeof(wifi_config.sta.ssid) - 1);
    std::strncpy(reinterpret_cast<char *>(wifi_config.sta.password), WIFI_PASSWORD,
                 sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    wifi_config.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi SSID \"%s\"...", WIFI_SSID);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
esp_err_t http_image_server_start()
{
    esp_err_t err = wifi_init_sta();
    if (err != ESP_OK) {
        return err;
    }
    return start_httpd() ? ESP_OK : ESP_FAIL;
}

#include <array>

#include "esp_log.h"
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "led_strip.h"
#include "led_strip_rmt.h"
#include "dl_model_base.hpp"
#include "dl_image_preprocessor.hpp"

#include "http_image_server.hpp"

// Camera pin map for ESP32-S3-WROOM-1 N16R8 CAM boards
// (CAMERA_MODEL_ESP32S3_EYE / Keyestudio MB0184 / OceanLabz N16R8).
#define CAM_PIN_PWDN    -1 //power down is not used
#define CAM_PIN_RESET   -1 //software reset will be performed
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD     4
#define CAM_PIN_SIOC     5

#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2       8
#define CAM_PIN_D1       9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC    6
#define CAM_PIN_HREF     7
#define CAM_PIN_PCLK    13

// Onboard addressable RGB LED (WS2812) data line.
#define LED_STRIP_GPIO       48
#define LED_STRIP_LED_COUNT  1

static const char *TAG = "MODEL";

static dl::Model *model = nullptr;
static dl::image::ImagePreprocessor *preprocessor = nullptr;
static led_strip_handle_t led_strip = nullptr;

static esp_err_t init_led()
{
    led_strip_config_t strip_config = {
        .strip_gpio_num = LED_STRIP_GPIO,
        .max_leds = LED_STRIP_LED_COUNT,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
        .flags = {.invert_out = false},
    };
    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 0,
        .flags = {.with_dma = false},
    };
    return led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip);
}

static void set_led(bool on)
{
    if (!led_strip) {
        return;
    }
    led_strip_set_pixel(led_strip, 0, on ? 32 : 0, 0, 0);
    led_strip_refresh(led_strip);
}

static void signal_ready()
{
    // Flash the onboard LED for ~3 s to indicate initialisation finished.
    for (int i = 0; i < 6; ++i) {
        set_led(true);
        vTaskDelay(pdMS_TO_TICKS(250));
        set_led(false);
        vTaskDelay(pdMS_TO_TICKS(250));
    }
}

static esp_err_t init_camera()
{
    camera_config_t config = {
        .pin_pwdn  = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,

        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,

        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        .pixel_format = PIXFORMAT_RGB565,
        .frame_size = FRAMESIZE_QVGA,

        .jpeg_quality = 5,
        // Two buffers so the HTTP handler and the detector can each hold a
        // frame; GRAB_LATEST keeps the preview fresh while the model runs.
        .fb_count = 2,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_LATEST,

        .sccb_i2c_port = 0,
        .jpeg_buffer_size = 0
    };
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera Init Failed");
        return err;
    }

    return ESP_OK;
}

static esp_err_t init_model()
{
    model = new dl::Model("model", fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    if (!model) {
        ESP_LOGE(TAG, "Model Init Failed");
        return ESP_FAIL;
    }

    if (model->test() == ESP_OK) {
        ESP_LOGI(TAG, "Model test passed!");
    } else {
        ESP_LOGE(TAG, "Model test failed");
    }

    // Training preprocessing is Resize((224,224)) -> ToTensor() -> x255, i.e. raw
    // RGB in [0, 255] with no mean/std shift. Matching that here means the
    // preprocessor converts the camera frame to RGB888, resizes, and quantizes with
    // the model input exponent (Q = round(pixel / 2^exponent)); it also writes
    // directly into the model input tensor in the NHWC layout the graph expects.
    // If colours come out swapped (and detection quality suffers), pass
    // rgb_swap = true as the 4th argument.
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> std  = {1.0f, 1.0f, 1.0f};
    preprocessor = new dl::image::ImagePreprocessor(model, mean, std, false);
    if (!preprocessor) {
        ESP_LOGE(TAG, "Image Preprocessor Init Failed");
        return ESP_FAIL;
    }

    return ESP_OK;
}

static void process()
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera Capture Failed");
        return;
    }
    if (fb->format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Unexpected pixel format: %d", (int)fb->format);
        esp_camera_fb_return(fb);
        return;
    }

    dl::image::img_t img = {
        .data = fb->buf,
        .width = static_cast<uint16_t>(fb->width),
        .height = static_cast<uint16_t>(fb->height),
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE,
    };

    // The preprocessor converts RGB565 -> RGB888, resizes the whole frame to the
    // model input size and quantizes it into the model input tensor, matching the
    // training-time Resize((224,224)) + [0,255] preprocessing.
    preprocessor->preprocess(img);

    model->run();

    dl::TensorBase *model_output = model->get_output();
    if (!model_output) {
        ESP_LOGE(TAG, "Model output is null");
        esp_camera_fb_return(fb);
        return;
    }

    // The output is INT8 with a single per-tensor exponent. Both logits share
    // that exponent, so comparing the raw int8 values is equivalent to comparing
    // the dequantized float logits (and avoids a lossy requantize).
    int8_t *res = static_cast<int8_t *>(model_output->data);

    if (res[1] >= res[0]) {
        ESP_LOGI(TAG, "Fire detected");
        set_led(true);
    } else {
        set_led(false);
    }

    esp_camera_fb_return(fb);
}

extern "C" void app_main(void)
{
    if (init_led() != ESP_OK) {
        ESP_LOGE(TAG, "LED Strip Init Failed");
    }
    set_led(false);

    if (init_camera() != ESP_OK) {
        return;
    }
    if (init_model() != ESP_OK) {
        return;
    }

#ifdef ENABLE_HTTP_IMAGE_SERVER
    if (http_image_server_start() != ESP_OK) {
        ESP_LOGE(TAG, "HTTP image server start failed");
    }
#endif

    // Everything is up: flash the LED so the board tells you it is ready.
    signal_ready();

    while (true) {
        process();
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

#pragma once

// WiFi station (STA) credentials.
//
// These are `inline constexpr` variables, not macros: `inline` gives each a
// single, linker-merged definition, so including this header from more than one
// translation unit still links (no "multiple definition" errors) and the
// symbols are present in the final ELF.
//
// Edit the values below. To keep the secret out of version control, add this
// file to .gitignore and ship a wifi_config.hpp.example instead.

inline constexpr char WIFI_SSID[] = "Ninh";
inline constexpr char WIFI_PASSWORD[] = "9fbc3597940c";

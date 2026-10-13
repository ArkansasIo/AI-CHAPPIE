#pragma once
#include <stdint.h>
namespace chappie::protocol {
constexpr uint8_t kVersion=1;
constexpr uint16_t kMaxPayload=128;
enum class MessageType:uint8_t {Heartbeat=1,Status=2,Command=3,Fault=4};
struct Header {uint8_t version; MessageType type; uint16_t length; uint32_t sequence; uint32_t timestamp_ms;};
// TODO: define byte-order serialization and CRC. Never reinterpret raw wire bytes as this struct.
}

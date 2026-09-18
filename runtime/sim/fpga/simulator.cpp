/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ftdi.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "hw_sim/coralnpu_simulator.h"
#include "sw/utils/nexus_loader/spi_master.h"

namespace {

// FTDI USB Vendor ID.
constexpr uint16_t kFtdiVid = 0x0403;
// FT4232H Quad High-Speed USB-to-MPSSE bridge Product ID.
constexpr uint16_t kFtdiPid = 0x6011;
// CoralNPU core CSR base address in the high-memory SoC crossbar map
// (+0x0 reset, +0x4 start PC, +0x8 halt status).
constexpr uint32_t kCsrBase = 0x200000;

struct FtdiImpl : public FtdiInterface {
  struct ftdi_context* ftdi = nullptr;

  FtdiImpl() { ftdi = ftdi_new(); }
  ~FtdiImpl() override {
    if (ftdi) {
      close_usb();
      ftdi_free(ftdi);
    }
  }

  void close_usb() {
    if (ftdi && ftdi->usb_dev != nullptr) {
      if (ftdi_usb_close(ftdi) < 0) {
        std::fprintf(stderr,
                     "[CoralNPU FPGA] Error: ftdi_usb_close failed: %s\n",
                     ftdi_get_error_string(ftdi));
      }
    }
  }

  int write_data(const uint8_t* buf, int size) override {
    return ftdi ? ftdi_write_data(ftdi, buf, size) : -1;
  }

  int read_data(uint8_t* buf, int size) override {
    return ftdi ? ftdi_read_data(ftdi, buf, size) : -1;
  }

  int purge_buffers() override { return ftdi ? ftdi_tcioflush(ftdi) : -1; }
};

class FpgaSimulator final : public CoralNPUSimulator {
 public:
  FpgaSimulator() = default;
  ~FpgaSimulator() override = default;

  bool Connect(const char* serial) {
    if (connected_) return true;
    if (!impl_.ftdi) {
      std::fprintf(stderr, "[CoralNPU FPGA] Error: ftdi_new failed.\n");
      return false;
    }

    if (ftdi_set_interface(impl_.ftdi, INTERFACE_A) < 0) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: ftdi_set_interface failed: %s\n",
                   ftdi_get_error_string(impl_.ftdi));
      return false;
    }
    if (ftdi_usb_open_desc(impl_.ftdi, kFtdiVid, kFtdiPid, nullptr,
                           serial && *serial ? serial : nullptr) < 0) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: ftdi_usb_open_desc failed: %s\n",
                   ftdi_get_error_string(impl_.ftdi));
      return false;
    }

    if (ftdi_set_bitmode(impl_.ftdi, 0, BITMODE_RESET) < 0) {
      std::fprintf(
          stderr,
          "[CoralNPU FPGA] Error: ftdi_set_bitmode (RESET) failed: %s\n",
          ftdi_get_error_string(impl_.ftdi));
      impl_.close_usb();
      return false;
    }
    if (ftdi_set_bitmode(impl_.ftdi, kDirMask, BITMODE_MPSSE) < 0) {
      std::fprintf(
          stderr,
          "[CoralNPU FPGA] Error: ftdi_set_bitmode (MPSSE) failed: %s\n",
          ftdi_get_error_string(impl_.ftdi));
      impl_.close_usb();
      return false;
    }

    const uint8_t init_cmd[] = {
        MPSSE_DISABLE_CLK_DIV_5,
        MPSSE_ENABLE_3PHASE_CLK,
        MPSSE_SET_TCK_DIVISOR,
        0x00,
        0x00,
        MPSSE_DISABLE_3PHASE_CLK,
        MPSSE_SET_DATA_BITS_LOW,
        kCsHigh,
        kDirMask,
    };
    if (impl_.write_data(init_cmd, sizeof(init_cmd)) !=
        static_cast<int>(sizeof(init_cmd))) {
      std::fprintf(
          stderr,
          "[CoralNPU FPGA] Error: failed to write MPSSE init command: %s\n",
          ftdi_get_error_string(impl_.ftdi));
      impl_.close_usb();
      return false;
    }

    connected_ = true;
    return true;
  }

  void ReadMem(uint32_t addr, size_t size, char* data) override {
    if (!connected_) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: Cannot read memory: "
                   "device not connected.\n");
      if (data && size > 0) {
        std::memset(data, 0, size);
      }
      return;
    }
    if (!spi_.v2_read_data(addr, size, reinterpret_cast<uint8_t*>(data))) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: ReadMem failed at 0x%08x "
                   "(size %zu).\n",
                   addr, size);
      if (data && size > 0) {
        std::memset(data, 0, size);
      }
    }
  }

  const CoralNPUMailbox& ReadMailbox(void) override {
    static CoralNPUMailbox dummy{};
    return dummy;
  }

  void WriteMem(uint32_t addr, size_t size, const char* data) override {
    if (!connected_) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: Cannot write memory: "
                   "device not connected.\n");
      return;
    }
    spi_.v2_write_data(addr, reinterpret_cast<const uint8_t*>(data), size);
  }

  void WriteMailbox(const CoralNPUMailbox& mailbox) override {}

  void Run(uint32_t start_addr) override {
    if (!connected_) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: Cannot run: "
                   "device not connected.\n");
      return;
    }
    // Set start PC, pulse reset, release
    spi_.write_word(kCsrBase + 4, start_addr);
    spi_.write_word(kCsrBase, 1);
    if (usleep(1000) < 0) {
      std::fprintf(stderr, "[CoralNPU FPGA] Warning: usleep failed: %s\n",
                   std::strerror(errno));
    }
    spi_.write_word(kCsrBase, 0);
  }

  // |timeout| counts simulator steps, so bound the wall clock instead.
  bool WaitForTermination(int /*timeout*/) override {
    if (!connected_) {
      std::fprintf(stderr,
                   "[CoralNPU FPGA] Error: Cannot wait for termination: "
                   "device not connected.\n");
      return false;
    }

    constexpr double timeout_sec = 30.0;
    auto start = std::chrono::steady_clock::now();
    int retries = 0;
    constexpr int kMaxRetries = 10;

    while (true) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start)
              .count() > timeout_sec) {
        std::fprintf(stderr,
                     "[CoralNPU FPGA] Execution timed out waiting for halt.\n");
        return false;
      }
      uint32_t val = 0;
      if (!spi_.v2_read_data(kCsrBase + 8, 4,
                             reinterpret_cast<uint8_t*>(&val))) {
        retries++;
        if (retries > kMaxRetries) {
          std::fprintf(stderr,
                       "[CoralNPU FPGA] Error: Failed to read halt status "
                       "after %d retries.\n",
                       kMaxRetries);
          return false;
        }
        if (usleep(10000) < 0) {
          std::fprintf(stderr, "[CoralNPU FPGA] Warning: usleep failed: %s\n",
                       std::strerror(errno));
        }
        continue;
      }
      retries = 0;
      if (val == 1) {
        return true;
      }
      if (usleep(10000) < 0) {
        std::fprintf(stderr, "[CoralNPU FPGA] Warning: usleep failed: %s\n",
                     std::strerror(errno));
      }
    }
  }

  uint64_t GetCycleCount() const override { return 0; }

 private:
  FtdiImpl impl_;
  SpiMaster spi_{&impl_};
  bool connected_ = false;
};

}  // namespace

extern "C" __attribute__((visibility("default"))) CoralNPUSimulator*
coralnpu_simulator_fpga_create(void) {
  auto sim = std::make_unique<FpgaSimulator>();
  const char* serial = std::getenv("CORALNPU_FPGA_SERIAL");
  if (!serial || !*serial) {
    std::fprintf(stderr,
                 "[CoralNPU FPGA] Error: set CORALNPU_FPGA_SERIAL to the "
                 "board's FTDI serial (e.g. Nexus-FTDI-12).\n");
    return nullptr;
  }
  if (!sim->Connect(serial)) {
    std::fprintf(stderr,
                 "[CoralNPU FPGA] Notice: FPGA device not connected "
                 "(serial: %s).\n",
                 serial ? serial : "(auto)");
    return nullptr;
  }
  return sim.release();
}

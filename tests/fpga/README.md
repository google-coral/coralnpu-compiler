# CoralNPU Nexus FPGA Hardware Tests

This directory contains NPU-only compilation and FPGA hardware execution targets for Nexus FPGA boards connected via FTDI USB/SPI.

Tests are split into two categories:
- **`check` tests (`check_tests/*.vmfb`)**: Self-contained `check`-dialect modules executed via `iree-check-module` (no external inputs needed).
- **Models (`models/*.vmfb`)**: Standard modules (such as MobileNetV2) executed via `iree-run-module` with explicit `--function` and `--input` flags.

---

## 1. Running on a Remote FPGA Host (Standalone Tarball)

### Step 1: Build the Tarball
From the repository root, build the self-contained archive:

```bash
# Default (check_tests only):
bazel build --config=dev //tests/fpga:fpga_tests_tar

# Include models (models/mobilenet_v2.vmfb):
bazel build --config=dev --//tests/fpga:include_models //tests/fpga:fpga_tests_tar
```

This creates `bazel-bin/tests/fpga/fpga_tests.tar.gz` containing the statically
linked `iree-check-module` (`//tools:iree-check-module-static`) and
`iree-run-module` (`//tools:iree-run-module-static`) binaries (with the FPGA and
MPACT backends built in) and the compiled `.vmfb` modules:

```text
fpga_tests.tar.gz
├── bin/
│   ├── iree-check-module
│   └── iree-run-module
├── check_tests/
│   ├── add_i32.vmfb
│   ├── matmul.vmfb
│   ├── matmul_add.vmfb
│   ├── matmul_weight.vmfb
│   ├── medium_10dispatch_net.vmfb
│   └── mini_multi_dispatch.vmfb
└── models/
    └── mobilenet_v2.vmfb
```

### Step 2: Deploy, Extract & Set Environment Variables

```bash
rsync -avz bazel-bin/tests/fpga/fpga_tests.tar.gz <username>@<fpga-host-machine>:~/
ssh <username>@<fpga-host-machine>
mkdir -p ~/fpga_tests
tar -xzf ~/fpga_tests.tar.gz -C ~/fpga_tests
cd ~/fpga_tests

export CORALNPU_FPGA_SERIAL=<YOUR_FTDI_SERIAL>
```

### Step 3: Run the `iree-check-module` Tests
Run all self-contained `check_tests/*.vmfb` modules:

```bash
for m in check_tests/*.vmfb; do
  echo "=== Running $m ==="
  ./bin/iree-check-module --device=coralnpu --simulator=fpga --module="$m"
done
```

Or run an individual check module directly:

```bash
./bin/iree-check-module \
  --device=coralnpu \
  --simulator=fpga \
  --module=check_tests/medium_10dispatch_net.vmfb
```

### Step 4: Run Models Manually with Inputs (`iree-run-module`)
Run MobileNetV2 (`models/mobilenet_v2.vmfb`):

```bash
./bin/iree-run-module \
  --device=coralnpu \
  --simulator=fpga \
  --module=models/mobilenet_v2.vmfb \
  --function=main \
  --input=1x224x224x3xf32=0
```

---

## 2. Running Locally via Bazel

If the Nexus board is attached directly to your local workstation, run the targets via Bazel:

```bash
# Run the check-dialect hardware tests (runs locally and sequentially):
bazel test --config=dev --test_env=CORALNPU_FPGA_SERIAL=<YOUR_FTDI_SERIAL> //tests/fpga:npu_fpga_tests

# Run the MobileNetV2 hardware test:
bazel test --config=dev --test_env=CORALNPU_FPGA_SERIAL=<YOUR_FTDI_SERIAL> //tests/fpga:mobilenet_v2_fpga_test
```

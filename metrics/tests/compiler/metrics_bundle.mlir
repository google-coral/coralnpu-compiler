// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// --coralnpu-roofline-speedup-threshold=0 places the small matmul on CoralNPU;
// by default, the roofline model keeps it on the host.
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=pretty %s -o /dev/null 2>&1 >/dev/null | FileCheck %s --check-prefix=CHECK-PRETTY
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=json --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s --check-prefix=CHECK-JSON
// RUN: rm -rf %t && %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=json --coralnpu-dump-metrics-dir=%t %s -o /dev/null && cat %t/main_dispatch_0.metrics.json | FileCheck %s --check-prefix=CHECK-JSON
// RUN: rm -rf %t && %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=pretty --coralnpu-dump-metrics-dir=%t %s -o /dev/null && cat %t/main_dispatch_0.metrics.txt | FileCheck %s --check-prefix=CHECK-PRETTY
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NONE --allow-empty
// RUN: not %iree_compile --coralnpu-dump-metrics-format=csv %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-BAD

// CHECK-PRETTY:      ========================================================================
// CHECK-PRETTY-NEXT: CoralNPU Metrics: main_dispatch_0
// CHECK-PRETTY-NEXT: ========================================================================
// CHECK-PRETTY-NEXT: Target:
// CHECK-PRETTY-NEXT:   Triple: riscv32{{.*}}
// CHECK-PRETTY-NEXT:   CPU Features: {{.*}}+zvl128b{{.*}}
// CHECK-PRETTY-NEXT:   ABI: ilp32
// CHECK-PRETTY-NEXT:   VLEN: 128
// CHECK-PRETTY-NEXT:   Float: Yes, Zvt: Yes
// CHECK-PRETTY-NEXT:   DTCM (KB): 32, Vector Registers: 32
// CHECK-PRETTY-NEXT: Dispatches: 1
// CHECK-PRETTY-NEXT:   Dispatch: main_dispatch_0_matmul_16x16x16_i32
// CHECK-PRETTY-NEXT:     Ordinal: 0
// CHECK-PRETTY-NEXT:     Workgroup Size: 1x1x1
// CHECK-PRETTY-NEXT:     Bindings: 3, Constants: 0

// CHECK-JSON:      {
// CHECK-JSON-NEXT:   "schema": "coralnpu.metrics.v1",
// CHECK-JSON-NEXT:   "executable": "main_dispatch_0",
// CHECK-JSON-NEXT:   "target": {
// CHECK-JSON-NEXT:     "triple": "riscv32{{.*}}",
// CHECK-JSON-NEXT:     "cpu_features": "{{.*}}+zvl128b{{.*}}",
// CHECK-JSON-NEXT:     "abi": "ilp32",
// CHECK-JSON-NEXT:     "vlen": 128,
// CHECK-JSON-NEXT:     "float": true,
// CHECK-JSON-NEXT:     "zvt": true,
// CHECK-JSON-NEXT:     "dtcm_kb": 32,
// CHECK-JSON-NEXT:     "num_vector_registers": 32
// CHECK-JSON-NEXT:   },
// CHECK-JSON-NEXT:   "dispatches": [
// CHECK-JSON-NEXT:     {
// CHECK-JSON-NEXT:       "name": "main_dispatch_0_matmul_16x16x16_i32",
// CHECK-JSON-NEXT:       "ordinal": 0,
// CHECK-JSON-NEXT:       "workgroup_size": [
// CHECK-JSON-NEXT:         1,
// CHECK-JSON-NEXT:         1,
// CHECK-JSON-NEXT:         1
// CHECK-JSON-NEXT:       ],
// CHECK-JSON-NEXT:       "binding_count": 3,
// CHECK-JSON-NEXT:       "constant_count": 0
// CHECK-JSON-NEXT:     }
// CHECK-JSON-NEXT:   ],
// CHECK-JSON-NEXT:   "elf": {

// CHECK-NONE-NOT: CoralNPU Metrics
// CHECK-NONE-NOT: coralnpu.metrics.v1

// CHECK-BAD: Cannot find option named 'csv'

func.func @main(
    %arg0: tensor<16x16xi32>,
    %arg1: tensor<16x16xi32>)
    -> (tensor<16x16xi32>) {
  %matmul = stablehlo.dot %arg0, %arg1
      : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
  return %matmul : tensor<16x16xi32>
}

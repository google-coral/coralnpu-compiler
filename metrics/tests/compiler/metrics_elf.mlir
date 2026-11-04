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

// ELF size report of the metrics bundle. The RUN lines use
// --coralnpu-roofline-speedup-threshold=0 to place the small dispatch on
// CoralNPU; by default, the roofline model keeps it on the host.

// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=json --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s --check-prefix=CHECK-JSON
// RUN: %iree_compile --coralnpu-roofline-speedup-threshold=0 --coralnpu-dump-metrics-format=pretty --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s --check-prefix=CHECK-PRETTY

// CHECK-JSON:        "elf": {
// CHECK-JSON-NEXT:     "size_bytes": {{[1-9][0-9]*}},
// CHECK-JSON-NEXT:     "regions": {
// CHECK-JSON-NEXT:       "itcm": {
// CHECK-JSON-NEXT:         "start": 0,
// CHECK-JSON-NEXT:         "size": 8192,
// CHECK-JSON-NEXT:         "used": {{[1-9][0-9]*}},
// CHECK-JSON-NEXT:         "free": {{[1-9][0-9]*}}
// CHECK-JSON-NEXT:       },
// CHECK-JSON-NEXT:       "dtcm": {
// CHECK-JSON-NEXT:         "start": 65536,
// CHECK-JSON-NEXT:         "size": 32768,
// CHECK-JSON-NEXT:         "used": {{[1-9][0-9]*}},
// CHECK-JSON-NEXT:         "stack": 4096,
// CHECK-JSON-NEXT:         "free": [[DTCM_FREE:[1-9][0-9]*]]
// CHECK-JSON-NEXT:       }
// CHECK-JSON-NEXT:     },
// CHECK-JSON-NEXT:     "sections": [
// CHECK-JSON-NEXT:       {
// CHECK-JSON-NEXT:         "name": ".text",
// CHECK-JSON-NEXT:         "region": "itcm",
// CHECK-JSON-NEXT:         "address": 0,
// CHECK-JSON-NEXT:         "size": {{[1-9][0-9]*}},
// CHECK-JSON-NEXT:         "type": "progbits"
// CHECK-JSON-NEXT:       },
// CHECK-JSON:              "name": ".heap",
// CHECK-JSON-NEXT:         "region": "dtcm",
// CHECK-JSON-NEXT:         "address": {{[0-9]+}},
// CHECK-JSON-NEXT:         "size": [[DTCM_FREE]],
// CHECK-JSON-NEXT:         "type": "nobits"
// CHECK-JSON:              "name": ".stack",
// CHECK-JSON-NEXT:         "region": "dtcm",
// CHECK-JSON-NEXT:         "address": {{[0-9]+}},
// CHECK-JSON-NEXT:         "size": 4096,
// CHECK-JSON-NEXT:         "type": "nobits"
// CHECK-JSON-NEXT:       }
// CHECK-JSON-NEXT:     ]
// CHECK-JSON-NEXT:   }
// CHECK-JSON-NEXT: }

// CHECK-PRETTY:      ELF:
// CHECK-PRETTY-NEXT:   File Size (bytes): {{[1-9][0-9]*}}
// CHECK-PRETTY-NEXT:   Regions:
// CHECK-PRETTY-NEXT:     ITCM  {{ *}}{{[0-9]+}} /     8192 bytes ({{ *[0-9.]+}}%), free {{[0-9]+}}
// CHECK-PRETTY-NEXT:     DTCM  {{ *}}{{[0-9]+}} /    32768 bytes ({{ *[0-9.]+}}%), stack 4096, free {{[0-9]+}}
// CHECK-PRETTY-NEXT:   Sections:
// CHECK-PRETTY-NEXT:     Name            Region  Address         Size  Type
// CHECK-PRETTY-NEXT:     .text           itcm    0x00000000 {{ *[0-9]+}}  progbits
// CHECK-PRETTY:          .stack          dtcm    0x{{[0-9a-f]+}} {{ *}}4096  nobits

func.func @main(
    %arg0: tensor<16x16xi32>,
    %arg1: tensor<16x16xi32>)
    -> (tensor<16x16xi32>) {
  %matmul = stablehlo.dot %arg0, %arg1
      : (tensor<16x16xi32>, tensor<16x16xi32>) -> tensor<16x16xi32>
  return %matmul : tensor<16x16xi32>
}

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

// RUN: %coralnpu_compile --coralnpu-dump-register-allocation-report-format=pretty --coralnpu-dump-register-allocation-report-dir=- %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-PRETTY
// RUN: %coralnpu_compile --coralnpu-dump-register-allocation-report-format=json --coralnpu-dump-register-allocation-report-dir=- %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-JSON

// CHECK-PRETTY: ========================================================================
// CHECK-PRETTY: Register Allocation Report:
// CHECK-PRETTY: ========================================================================
// CHECK-PRETTY: Dispatch: main_dispatch_0_matmul_128x128x128_i32
// CHECK-PRETTY:   Total Spills: 20, Total Reloads: 20, Total Copies: 9
// CHECK-PRETTY:   Function-level (non-loop):
// CHECK-PRETTY:     Location: {{.*}}
// CHECK-PRETTY:     Vector Registers Used: 16 [V10M2, V12M2, V14M2, V16M2, V18M2, V20M2, V22M2, V8M2]
// CHECK-PRETTY:   Loop (depth 1) at %bb.1 ({{.*}}):
// CHECK-PRETTY:     Vector Registers Used: 16 [V10M2, V12M2, V14M2, V16M2, V18M2, V20M2, V22M2, V8M2]
// CHECK-PRETTY:     Spills: 5, Reloads: 20, Copies: 7
// CHECK-PRETTY:   Loop (depth 2) at %bb.2 ({{.*}}):
// CHECK-PRETTY:     Vector Registers Used: 14 [V10M2, V12M2, V14M2, V16M2, V18M2, V20M2, V22M2]
// CHECK-PRETTY:     Spills: 0, Reloads: 7, Copies: 6

// CHECK-JSON: {
// CHECK-JSON:   "dispatches": [
// CHECK-JSON:     {
// CHECK-JSON:       "name": "main_dispatch_0_matmul_128x128x128_i32",
// CHECK-JSON:       "spills": 20,
// CHECK-JSON:       "reloads": 20,
// CHECK-JSON:       "copies": 9,
// CHECK-JSON:       "global_location": "{{.*}}",
// CHECK-JSON:       "global_vector_registers_count": 16,
// CHECK-JSON:       "global_vector_registers": [
// CHECK-JSON:         "V10M2",
// CHECK-JSON:         "V12M2",
// CHECK-JSON:         "V14M2",
// CHECK-JSON:         "V16M2",
// CHECK-JSON:         "V18M2",
// CHECK-JSON:         "V20M2",
// CHECK-JSON:         "V22M2",
// CHECK-JSON:         "V8M2"
// CHECK-JSON:       ],
// CHECK-JSON:       "loops": [
// CHECK-JSON:         {
// CHECK-JSON:           "header": "%bb.1",
// CHECK-JSON:           "location": "{{.*}}",
// CHECK-JSON:           "depth": 1,
// CHECK-JSON:           "vector_registers_used_count": 16,
// CHECK-JSON:           "vector_registers_used": [
// CHECK-JSON:             "V10M2",
// CHECK-JSON:             "V12M2",
// CHECK-JSON:             "V14M2",
// CHECK-JSON:             "V16M2",
// CHECK-JSON:             "V18M2",
// CHECK-JSON:             "V20M2",
// CHECK-JSON:             "V22M2",
// CHECK-JSON:             "V8M2"
// CHECK-JSON:           ],
// CHECK-JSON:           "spills": 5,
// CHECK-JSON:           "reloads": 20,
// CHECK-JSON:           "copies": 7
// CHECK-JSON:         },
// CHECK-JSON:         {
// CHECK-JSON:           "header": "%bb.2",
// CHECK-JSON:           "location": "{{.*}}",
// CHECK-JSON:           "depth": 2,
// CHECK-JSON:           "vector_registers_used_count": 14,
// CHECK-JSON:           "vector_registers_used": [
// CHECK-JSON:             "V10M2",
// CHECK-JSON:             "V12M2",
// CHECK-JSON:             "V14M2",
// CHECK-JSON:             "V16M2",
// CHECK-JSON:             "V18M2",
// CHECK-JSON:             "V20M2",
// CHECK-JSON:             "V22M2"
// CHECK-JSON:           ],
// CHECK-JSON:           "spills": 0,
// CHECK-JSON:           "reloads": 7,
// CHECK-JSON:           "copies": 6
// CHECK-JSON:         }

func.func @main(
    %arg0: tensor<128x128xi32>,
    %arg1: tensor<128x128xi32>)
    -> (tensor<128x128xi32>) {
  %matmul = stablehlo.dot %arg0, %arg1
      : (tensor<128x128xi32>, tensor<128x128xi32>) -> tensor<128x128xi32>
  return %matmul : tensor<128x128xi32>
}

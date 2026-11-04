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

// Tests the `coralnpu.metrics` attribute channel: the entries that the metrics
// passes attach to a dispatch function are copied into the entry of the
// dispatch in the bundle, next to those of the metrics passes (tile_grid,
// zvt). Reserved keys and unsupported values are skipped.

// RUN: iree-compile --compile-mode=hal-executable --iree-hal-target-device=coralnpu --coralnpu-dump-metrics-format=json --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s
// RUN: iree-compile --compile-mode=hal-executable --iree-hal-target-device=coralnpu --coralnpu-dump-metrics-format=pretty --coralnpu-dump-metrics-dir=- %s -o /dev/null | FileCheck %s --check-prefix=CHECK-PRETTY
// RUN: iree-compile --compile-mode=hal-executable --iree-hal-target-device=coralnpu %s -o /dev/null

// CHECK:      "executable": "executable",
// CHECK:      "dispatches": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "name": "mul",
// CHECK-NEXT:     "ordinal": 0,
// CHECK:          "binding_count": 3,
// CHECK-NEXT:     "constant_count": 2,
// CHECK-NEXT:     "bytes": 48,
// CHECK-NEXT:     "enabled": true,
// CHECK-NEXT:     "macs": 4,
// CHECK-NEXT:     "nested": {
// CHECK-NEXT:       "shape": [
// CHECK-NEXT:         4
// CHECK-NEXT:       ],
// CHECK-NEXT:       "unit": true
// CHECK-NEXT:     },
// CHECK-NEXT:     "ratio": 0.5,
// CHECK-NEXT:     "tag": "test",
// CHECK-NEXT:     "tile_grid": {
// CHECK-NOT:      "nan_val"
// CHECK-NOT:      "unsupported"
// CHECK-NOT:      "wide_int"
// CHECK:          "unsigned_int": 7,
// CHECK-NEXT:     "zvt": {
// CHECK:        "elf": {

// CHECK-PRETTY:      Dispatch: mul
// CHECK-PRETTY:        Bindings: 3, Constants: 2
// CHECK-PRETTY-NEXT:   bytes: 48
// CHECK-PRETTY-NEXT:   enabled: true
// CHECK-PRETTY-NEXT:   macs: 4
// CHECK-PRETTY-NEXT:   nested: {"shape":[4],"unit":true}
// CHECK-PRETTY-NEXT:   ratio: 0.5
// CHECK-PRETTY-NEXT:   tag: "test"
// CHECK-PRETTY-NEXT:   tile_grid: {"dtcm":[4],"loops":[4],"op":"linalg.generic","vector":[4],"workgroup":[4]}
// CHECK-PRETTY-NEXT:   unsigned_int: 7
// CHECK-PRETTY-NEXT:   zvt: {"candidate":false,"lowered":false}
// CHECK-PRETTY-NEXT: ELF:

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @mul ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    hal.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @mul() attributes {
      coralnpu.metrics = {
        macs = 4 : i64,
        bytes = 48 : i64,
        ratio = 5.000000e-01 : f32,
        enabled = true,
        tag = "test",
        nested = {shape = [4 : i64], unit},
        unsigned_int = 7 : ui64,
        name = "reserved",
        nan_val = 0x7FC00000 : f32,
        wide_int = 1 : i128,
        unsupported = affine_map<(d0) -> (d0)>
      }
    } {
      %c0 = arith.constant 0 : index
      %s0b0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
      %s0b1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>>
      %s0b2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      %0 = iree_tensor_ext.dispatch.tensor.load %s0b0, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %1 = iree_tensor_ext.dispatch.tensor.load %s0b1, offsets = [0], sizes = [4], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %2 = tensor.empty() : tensor<4xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<4xf32>, tensor<4xf32>) outs(%2 : tensor<4xf32>) {
      ^bb0(%in0: f32, %in1: f32, %out: f32):
        %4 = arith.mulf %in0, %in1 : f32
        linalg.yield %4 : f32
      } -> tensor<4xf32>
      iree_tensor_ext.dispatch.tensor.store %3, %s0b2, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4xf32>>
      return
    }
  }
}

// RUN: %coralnpu_compile --coralnpu-dump-affinity-profile-format=pretty %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-PRETTY
// RUN: %coralnpu_compile --coralnpu-dump-affinity-profile-format=json %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-JSON
// RUN: %coralnpu_compile --coralnpu-dump-affinity-profile-format=csv %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-CSV

// CHECK-PRETTY: ========================================================================
// CHECK-PRETTY: Execution Profile by Affinity:
// CHECK-PRETTY: ========================================================================
// CHECK-PRETTY:   Affinity: coralnpu
// CHECK-PRETTY:     Dispatches: 1 (50.0%)
// CHECK-PRETTY:       Static:    1 (100.0%)
// CHECK-PRETTY:       Dynamic:   0 (0.0%)
// CHECK-PRETTY:     Estimated Static Data Size: 0.00 MB
// CHECK-PRETTY:     Estimated Static Elements: 0.00 M
// CHECK-PRETTY:     Estimated Static Work (Op-Bytes): 0.01 M (69.0%)
// CHECK-PRETTY:     Estimated Static Work (Op-Elems): 0.00 M (69.0%)
// CHECK-PRETTY:     Fills:  0
// CHECK-PRETTY:     Copies: 0
// CHECK-PRETTY:   Affinity: local
// CHECK-PRETTY:     Dispatches: 1 (50.0%)
// CHECK-PRETTY:       Static:    1 (100.0%)
// CHECK-PRETTY:       Dynamic:   0 (0.0%)
// CHECK-PRETTY:     Estimated Static Data Size: 0.00 MB
// CHECK-PRETTY:     Estimated Static Elements: 0.00 M
// CHECK-PRETTY:     Estimated Static Work (Op-Bytes): 0.00 M (31.0%)
// CHECK-PRETTY:     Estimated Static Work (Op-Elems): 0.00 M (31.0%)
// CHECK-PRETTY:     Fills:  0
// CHECK-PRETTY:     Copies: 0
// CHECK-PRETTY: ========================================================================

// CHECK-JSON: {
// CHECK-JSON:   "affinities": {
// CHECK-JSON:     "coralnpu": {
// CHECK-JSON:       "dispatch-count": 1,
// CHECK-JSON:       "static-dispatch-count": 1,
// CHECK-JSON:       "dynamic-dispatch-count": 0,
// CHECK-JSON:       "static-data-size-bytes": 320,
// CHECK-JSON:       "has-dynamic-data-size": false,
// CHECK-JSON:       "static-elements-count": 80,
// CHECK-JSON:       "has-dynamic-elements": false,
// CHECK-JSON:       "static-work-bytes": 5120,
// CHECK-JSON:       "static-work-elements": 1280,
// CHECK-JSON:       "fill-count": 0,
// CHECK-JSON:       "copy-count": 0
// CHECK-JSON:     },
// CHECK-JSON:     "local": {
// CHECK-JSON:       "dispatch-count": 1,
// CHECK-JSON:       "static-dispatch-count": 1,
// CHECK-JSON:       "dynamic-dispatch-count": 0,
// CHECK-JSON:       "static-data-size-bytes": 192,
// CHECK-JSON:       "has-dynamic-data-size": false,
// CHECK-JSON:       "static-elements-count": 48,
// CHECK-JSON:       "has-dynamic-elements": false,
// CHECK-JSON:       "static-work-bytes": 2304,
// CHECK-JSON:       "static-work-elements": 576,
// CHECK-JSON:       "fill-count": 0,
// CHECK-JSON:       "copy-count": 0
// CHECK-JSON:     }
// CHECK-JSON:   }
// CHECK-JSON: }

// CHECK-CSV: "Affinity","Dispatches","Static","Dynamic","Estimated Static Data Size (Bytes)","Estimated Static Elements","Estimated Static Work (Op-Bytes)","Estimated Static Work (Op-Elems)","Fills","Copies"
// CHECK-CSV: "coralnpu",1,1,0,320,80,5120,1280,0,0
// CHECK-CSV: "local",1,1,0,192,48,2304,576,0,0

func.func @main(
    %arg0: tensor<4x8xi32>,
    %arg1: tensor<8x4xi32>,
    %arg2: tensor<4x4xi32>,
    %arg3: tensor<4x4xi32>)
    -> (tensor<4x4xi32>, tensor<4x4xi32>) {
  %matmul = stablehlo.dot %arg0, %arg1 : (tensor<4x8xi32>, tensor<8x4xi32>) -> tensor<4x4xi32>
  %add = stablehlo.add %arg2, %arg3 : tensor<4x4xi32>
  return %matmul, %add : tensor<4x4xi32>, tensor<4x4xi32>
}

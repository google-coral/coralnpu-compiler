// RUN: %iree_compile --compile-to=preprocessing %s | FileCheck %s

// Device symbols here are synthetic; a real compile names globals @__device_N.

// CHECK-LABEL: @conv_with_coralnpu_affinity
func.func @conv_with_coralnpu_affinity(
    %input: tensor<1x113x113x32xf32>,
    %filter: tensor<3x3x32x64xf32>,
    %output: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  // CHECK: iree_linalg_ext.im2col
  // CHECK: linalg.generic
  // CHECK-NOT: linalg.conv_2d_nhwc_hwcf
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, stream.affinity = #hal.device.affinity<@__device_coralnpu>}
    ins(%input, %filter : tensor<1x113x113x32xf32>, tensor<3x3x32x64xf32>)
    outs(%output : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}

// CHECK-LABEL: @conv_without_coralnpu_affinity
func.func @conv_without_coralnpu_affinity(
    %input: tensor<1x113x113x32xf32>,
    %filter: tensor<3x3x32x64xf32>,
    %output: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  // CHECK-NOT: iree_linalg_ext.im2col
  // CHECK-NOT: linalg.generic
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, stream.affinity = #hal.device.affinity<@__device_other>}
    ins(%input, %filter : tensor<1x113x113x32xf32>, tensor<3x3x32x64xf32>)
    outs(%output : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %0 : tensor<1x56x56x64xf32>
}

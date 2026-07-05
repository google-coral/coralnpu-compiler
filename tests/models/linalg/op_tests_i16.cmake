# Generated from op_tests_i16.bzl. Do not edit directly.

op_tests(
  NAME "fill_rank1_i16"
  TEST "fill_rank1_i16.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "fill_rank2_i16"
  TEST "fill_rank2_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "fill_rank3_i16"
  TEST "fill_rank3_i16.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "fill_rank4_i16"
  TEST "fill_rank4_i16.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "generic_add_rank1_i16"
  TEST "generic_add_rank1_i16.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "generic_add_rank2_i16"
  TEST "generic_add_rank2_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "generic_multi_output_i16"
  TEST "generic_multi_output_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
  LABELS
    "manual"
)

op_tests(
  NAME "generic_add_rank3_i16"
  TEST "generic_add_rank3_i16.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "generic_add_rank4_i16"
  TEST "generic_add_rank4_i16.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "matmul_i16"
  TEST "matmul_i16.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300)"
    "(300,100)(100,450)"
)

op_tests(
  NAME "batch_matmul_i16"
  TEST "batch_matmul_i16.mlir"
  INSTANCES
    "(2,4,8)(2,8,4)"
    "(3,12,25)(3,25,30)"
)

op_tests(
  NAME "mmt4d_i16"
  TEST "mmt4d_i16.mlir"
  INSTANCES
    "(2,3,4,2)(3,3,2,2)"
    "(1,2,4,4)(2,2,4,4)"
)

op_tests(
  NAME "conv_2d_nhwc_hwcf_i16"
  TEST "conv_2d_nhwc_hwcf_i16.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4,4)"
    "(1,8,8,8)(3,3,8,16)"
)

op_tests(
  NAME "conv_2d_nhwc_hwcf_stride2_i16"
  TEST "conv_2d_nhwc_hwcf_stride2_i16.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4,4)"
    "(1,8,8,8)(3,3,8,16)"
)

op_tests(
  NAME "conv_2d_nhwc_hwcf_dilation2_i16"
  TEST "conv_2d_nhwc_hwcf_dilation2_i16.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4,4)"
    "(1,8,8,8)(3,3,8,16)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwc_i16"
  TEST "depthwise_conv_2d_nhwc_hwc_i16.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4)"
    "(1,8,8,8)(3,3,8)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwc_stride2_i16"
  TEST "depthwise_conv_2d_nhwc_hwc_stride2_i16.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4)"
    "(1,8,8,8)(3,3,8)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwc_dilation2_i16"
  TEST "depthwise_conv_2d_nhwc_hwc_dilation2_i16.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4)"
    "(1,8,8,8)(3,3,8)"
)

op_tests(
  NAME "depthwise_conv_2d_nchw_chw_i16"
  TEST "depthwise_conv_2d_nchw_chw_i16.mlir"
  INSTANCES
    "(1,4,6,6)(4,3,3)"
    "(1,8,8,8)(8,3,3)"
)

op_tests(
  NAME "generic_reduction_2d_i16"
  TEST "generic_reduction_2d_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "pack_i16"
  TEST "pack_i16.mlir"
  INSTANCES
    "(8,16)"
    "(16,32)"
    "(128,256)"
)

op_tests(
  NAME "pack_perm_i16"
  TEST "pack_perm_i16.mlir"
  INSTANCES
    "(8,16)"
    "(16,32)"
    "(128,256)"
)

op_tests(
  NAME "pack_padding_i16"
  TEST "pack_padding_i16.mlir"
  INSTANCES
    "(7,15)"
    "(17,33)"
    "(120,250)"
)

op_tests(
  NAME "unpack_i16"
  TEST "unpack_i16.mlir"
  INSTANCES
    "(1,1,8,16)"
    "(2,2,8,16)"
    "(16,16,8,16)"
)

op_tests(
  NAME "unpack_perm_i16"
  TEST "unpack_perm_i16.mlir"
  INSTANCES
    "(1,1,8,16)"
    "(2,2,8,16)"
    "(3,2,8,16)"
)

op_tests(
  NAME "map_i16"
  TEST "map_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "reduce_2d_i16"
  TEST "reduce_2d_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reduce_2d_dim0_i16"
  TEST "reduce_2d_dim0_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reduce_3d_dim1_2_i16"
  TEST "reduce_3d_dim1_2_i16.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "reduce_2d_max_i16"
  TEST "reduce_2d_max_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "transpose_2d_i16"
  TEST "transpose_2d_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "transpose_3d_i16"
  TEST "transpose_3d_i16.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "broadcast_i16"
  TEST "broadcast_i16.mlir"
  INSTANCES
    "(8)(4,8)"
    "(256)(120,256)"
    "(450)(300,450)"
)

op_tests(
  NAME "broadcast_dim1_i16"
  TEST "broadcast_dim1_i16.mlir"
  INSTANCES
    "(8)(8,4)"
    "(256)(256,120)"
    "(450)(450,300)"
)

op_tests(
  NAME "elementwise_add_i16"
  TEST "elementwise_add_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "add_i16"
  TEST "add_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "sub_i16"
  TEST "sub_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "mul_i16"
  TEST "mul_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "div_i16"
  TEST "div_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
  DEFAULT_GEN "tools_check_gen_generators_positive_sequential_vmfb"
)

op_tests(
  NAME "div_unsigned_i16"
  TEST "div_unsigned_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
  DEFAULT_GEN "tools_check_gen_generators_positive_sequential_vmfb"
)

op_tests(
  NAME "max_i16"
  TEST "max_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "min_i16"
  TEST "min_i16.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "select_i16"
  TEST "select_i16.mlir"
  INSTANCES
    "(4,8)(4,8)(4,8)"
    "(120,256)(120,256)(120,256)"
    "(300,450)(300,450)(300,450)"
  LABELS
    "manual"
)

op_tests(
  NAME "contract_matmul_i16"
  TEST "contract_matmul_i16.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300)"
    "(300,100)(100,450)"
)

op_tests(
  NAME "batch_reduce_matmul_i16"
  TEST "batch_reduce_matmul_i16.mlir"
  INSTANCES
    "(2,4,8)(2,8,4)"
    "(3,12,25)(3,25,30)"
)

op_tests(
  NAME "batch_mmt4d_i16"
  TEST "batch_mmt4d_i16.mlir"
  INSTANCES
    "(2,1,2,4,4)(2,2,2,4,4)"
    "(3,2,3,4,2)(3,3,3,2,2)"
)

op_tests(
  NAME "matvec_i16"
  TEST "matvec_i16.mlir"
  INSTANCES
    "(4,8)(8)"
    "(120,256)(256)"
    "(300,450)(450)"
)

op_tests(
  NAME "vecmat_i16"
  TEST "vecmat_i16.mlir"
  INSTANCES
    "(8)(8,4)"
    "(256)(256,120)"
    "(450)(450,300)"
)

op_tests(
  NAME "batch_matvec_i16"
  TEST "batch_matvec_i16.mlir"
  INSTANCES
    "(2,4,8)(2,8)"
    "(3,120,256)(3,256)"
)

op_tests(
  NAME "batch_vecmat_i16"
  TEST "batch_vecmat_i16.mlir"
  INSTANCES
    "(2,8)(2,8,4)"
    "(3,256)(3,256,120)"
)

op_tests(
  NAME "dot_i16"
  TEST "dot_i16.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "conv_1d_i16"
  TEST "conv_1d_i16.mlir"
  INSTANCES
    "(11)(3)"
    "(258)(5)"
    "(453)(4)"
)

op_tests(
  NAME "conv_1d_nwc_wcf_i16"
  TEST "conv_1d_nwc_wcf_i16.mlir"
  INSTANCES
    "(1,11,4)(3,4,8)"
    "(1,256,8)(5,8,16)"
)

op_tests(
  NAME "conv_1d_ncw_fcw_i16"
  TEST "conv_1d_ncw_fcw_i16.mlir"
  INSTANCES
    "(1,4,11)(8,4,3)"
    "(1,8,256)(16,8,5)"
)

op_tests(
  NAME "conv_2d_i16"
  TEST "conv_2d_i16.mlir"
  INSTANCES
    "(11,11)(3,3)"
    "(125,256)(5,7)"
)

op_tests(
  NAME "conv_3d_i16"
  TEST "conv_3d_i16.mlir"
  INSTANCES
    "(7,7,7)(3,3,3)"
    "(12,15,20)(2,3,4)"
)

op_tests(
  NAME "pooling_nhwc_sum_i16"
  TEST "pooling_nhwc_sum_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_sum_stride2_i16"
  TEST "pooling_nhwc_sum_stride2_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_sum_dilation2_i16"
  TEST "pooling_nhwc_sum_dilation2_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_i16"
  TEST "pooling_nhwc_max_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_stride2_i16"
  TEST "pooling_nhwc_max_stride2_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_dilation2_i16"
  TEST "pooling_nhwc_max_dilation2_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_min_i16"
  TEST "pooling_nhwc_min_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_unsigned_i16"
  TEST "pooling_nhwc_max_unsigned_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_min_unsigned_i16"
  TEST "pooling_nhwc_min_unsigned_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "depthwise_conv_1d_nwc_wc_i16"
  TEST "depthwise_conv_1d_nwc_wc_i16.mlir"
  INSTANCES
    "(1,11,4)(3,4)"
    "(1,256,8)(5,8)"
)

op_tests(
  NAME "depthwise_conv_1d_ncw_cw_i16"
  TEST "depthwise_conv_1d_ncw_cw_i16.mlir"
  INSTANCES
    "(1,4,11)(4,3)"
    "(1,8,256)(8,5)"
)

op_tests(
  NAME "depthwise_conv_1d_nwc_wcm_i16"
  TEST "depthwise_conv_1d_nwc_wcm_i16.mlir"
  INSTANCES
    "(1,11,4)(3,4,2)"
    "(1,256,8)(5,8,1)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwcm_i16"
  TEST "depthwise_conv_2d_nhwc_hwcm_i16.mlir"
  INSTANCES
    "(1,11,11,4)(3,3,4,2)"
    "(1,120,256,8)(5,5,8,1)"
)

op_tests(
  NAME "depthwise_conv_3d_ndhwc_dhwc_i16"
  TEST "depthwise_conv_3d_ndhwc_dhwc_i16.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3,4)"
    "(1,12,15,20,8)(2,3,4,8)"
)

op_tests(
  NAME "conv_2d_nchw_fchw_i16"
  TEST "conv_2d_nchw_fchw_i16.mlir"
  INSTANCES
    "(1,4,6,6)(4,4,3,3)"
    "(1,8,8,8)(16,8,3,3)"
)

op_tests(
  NAME "conv_2d_nhwc_fhwc_i16"
  TEST "conv_2d_nhwc_fhwc_i16.mlir"
  INSTANCES
    "(1,6,6,4)(4,3,3,4)"
    "(1,8,8,8)(16,3,3,8)"
)

op_tests(
  NAME "pooling_nchw_sum_i16"
  TEST "pooling_nchw_sum_i16.mlir"
  INSTANCES
    "(1,4,11,11)(3,3)"
    "(1,8,120,256)(5,7)"
)

op_tests(
  NAME "pooling_nchw_max_i16"
  TEST "pooling_nchw_max_i16.mlir"
  INSTANCES
    "(1,4,11,11)(3,3)"
    "(1,8,120,256)(5,7)"
)

op_tests(
  NAME "pooling_nwc_sum_i16"
  TEST "pooling_nwc_sum_i16.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_nwc_max_i16"
  TEST "pooling_nwc_max_i16.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_nwc_min_i16"
  TEST "pooling_nwc_min_i16.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_nwc_max_unsigned_i16"
  TEST "pooling_nwc_max_unsigned_i16.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_nwc_min_unsigned_i16"
  TEST "pooling_nwc_min_unsigned_i16.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_ncw_sum_i16"
  TEST "pooling_ncw_sum_i16.mlir"
  INSTANCES
    "(1,4,11)(3)"
    "(1,8,120)(5)"
)

op_tests(
  NAME "pooling_ncw_max_i16"
  TEST "pooling_ncw_max_i16.mlir"
  INSTANCES
    "(1,4,11)(3)"
    "(1,8,120)(5)"
)

op_tests(
  NAME "pooling_ndhwc_sum_i16"
  TEST "pooling_ndhwc_sum_i16.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3)"
    "(1,12,15,20,8)(2,3,4)"
)

op_tests(
  NAME "pooling_ndhwc_max_i16"
  TEST "pooling_ndhwc_max_i16.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3)"
    "(1,12,15,20,8)(2,3,4)"
)

op_tests(
  NAME "pooling_ndhwc_min_i16"
  TEST "pooling_ndhwc_min_i16.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3)"
    "(1,12,15,20,8)(2,3,4)"
)

op_tests(
  NAME "copy_i16"
  TEST "copy_i16.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

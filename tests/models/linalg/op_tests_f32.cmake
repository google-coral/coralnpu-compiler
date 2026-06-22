# Generated from op_tests_f32.bzl. Do not edit directly.

op_tests(
  NAME "fill_rank1_f32"
  TEST "fill_rank1_f32.mlir"
  INSTANCES
    "(8)"
    "(256)"
    "(450)"
)

op_tests(
  NAME "fill_rank2_f32"
  TEST "fill_rank2_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "fill_rank3_f32"
  TEST "fill_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "fill_rank4_f32"
  TEST "fill_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)"
    "(2,3,4,50)"
    "(1,1,5,400)"
)

op_tests(
  NAME "generic_add_rank1_f32"
  TEST "generic_add_rank1_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "generic_add_rank2_f32"
  TEST "generic_add_rank2_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "generic_multi_output_f32"
  TEST "generic_multi_output_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
  LABELS
    "manual"
)

op_tests(
  NAME "generic_add_rank3_f32"
  TEST "generic_add_rank3_f32.mlir"
  INSTANCES
    "(2,3,4)(2,3,4)"
    "(10,20,30)(10,20,30)"
    "(5,100,2)(5,100,2)"
)

op_tests(
  NAME "generic_add_rank4_f32"
  TEST "generic_add_rank4_f32.mlir"
  INSTANCES
    "(2,2,3,2)(2,2,3,2)"
    "(2,3,4,50)(2,3,4,50)"
    "(1,1,5,400)(1,1,5,400)"
)

op_tests(
  NAME "matmul_f32"
  TEST "matmul_f32.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300)"
    "(300,100)(100,450)"
)

op_tests(
  NAME "batch_matmul_f32"
  TEST "batch_matmul_f32.mlir"
  INSTANCES
    "(2,4,8)(2,8,4)"
    "(3,12,25)(3,25,30)"
)

op_tests(
  NAME "mmt4d_f32"
  TEST "mmt4d_f32.mlir"
  INSTANCES
    "(2,3,4,2)(3,3,2,2)"
    "(1,2,4,4)(2,2,4,4)"
)

op_tests(
  NAME "conv_2d_nhwc_hwcf_f32"
  TEST "conv_2d_nhwc_hwcf_f32.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4,4)"
    "(1,8,8,8)(3,3,8,16)"
)

op_tests(
  NAME "conv_2d_nhwc_hwcf_stride2_f32"
  TEST "conv_2d_nhwc_hwcf_stride2_f32.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4,4)"
    "(1,8,8,8)(3,3,8,16)"
)

op_tests(
  NAME "conv_2d_nhwc_hwcf_dilation2_f32"
  TEST "conv_2d_nhwc_hwcf_dilation2_f32.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4,4)"
    "(1,8,8,8)(3,3,8,16)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwc_f32"
  TEST "depthwise_conv_2d_nhwc_hwc_f32.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4)"
    "(1,8,8,8)(3,3,8)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwc_stride2_f32"
  TEST "depthwise_conv_2d_nhwc_hwc_stride2_f32.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4)"
    "(1,8,8,8)(3,3,8)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwc_dilation2_f32"
  TEST "depthwise_conv_2d_nhwc_hwc_dilation2_f32.mlir"
  INSTANCES
    "(1,6,6,4)(3,3,4)"
    "(1,8,8,8)(3,3,8)"
)

op_tests(
  NAME "depthwise_conv_2d_nchw_chw_f32"
  TEST "depthwise_conv_2d_nchw_chw_f32.mlir"
  INSTANCES
    "(1,4,6,6)(4,3,3)"
    "(1,8,8,8)(8,3,3)"
)

op_tests(
  NAME "generic_reduction_2d_f32"
  TEST "generic_reduction_2d_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "pack_f32"
  TEST "pack_f32.mlir"
  INSTANCES
    "(8,16)"
    "(16,32)"
    "(128,256)"
)

op_tests(
  NAME "pack_perm_f32"
  TEST "pack_perm_f32.mlir"
  INSTANCES
    "(8,16)"
    "(16,32)"
    "(128,256)"
)

op_tests(
  NAME "pack_padding_f32"
  TEST "pack_padding_f32.mlir"
  INSTANCES
    "(7,15)"
    "(17,33)"
    "(120,250)"
)

op_tests(
  NAME "unpack_f32"
  TEST "unpack_f32.mlir"
  INSTANCES
    "(1,1,8,16)"
    "(2,2,8,16)"
    "(16,16,8,16)"
)

op_tests(
  NAME "unpack_perm_f32"
  TEST "unpack_perm_f32.mlir"
  INSTANCES
    "(1,1,8,16)"
    "(2,2,8,16)"
    "(3,2,8,16)"
)

op_tests(
  NAME "map_f32"
  TEST "map_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "reduce_2d_f32"
  TEST "reduce_2d_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reduce_2d_dim0_f32"
  TEST "reduce_2d_dim0_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reduce_3d_dim1_2_f32"
  TEST "reduce_3d_dim1_2_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "reduce_2d_max_f32"
  TEST "reduce_2d_max_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "transpose_2d_f32"
  TEST "transpose_2d_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "transpose_3d_f32"
  TEST "transpose_3d_f32.mlir"
  INSTANCES
    "(2,3,4)"
    "(10,20,30)"
    "(5,100,2)"
)

op_tests(
  NAME "broadcast_f32"
  TEST "broadcast_f32.mlir"
  INSTANCES
    "(8)(4,8)"
    "(256)(120,256)"
    "(450)(300,450)"
)

op_tests(
  NAME "broadcast_dim1_f32"
  TEST "broadcast_dim1_f32.mlir"
  INSTANCES
    "(8)(8,4)"
    "(450)(450,300)"
)

op_tests(
  NAME "broadcast_dim1_f32_bug"
  TEST "broadcast_dim1_f32.mlir"
  INSTANCES
    "(256)(256,120)"
  LABELS
    "manual"
)

op_tests(
  NAME "elementwise_add_f32"
  TEST "elementwise_add_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "abs_f32"
  TEST "abs_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "exp_f32"
  TEST "exp_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "log_f32"
  TEST "log_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "ceil_f32"
  TEST "ceil_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "floor_f32"
  TEST "floor_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "round_f32"
  TEST "round_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "sqrt_f32"
  TEST "sqrt_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "rsqrt_f32"
  TEST "rsqrt_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "square_f32"
  TEST "square_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "tanh_f32"
  TEST "tanh_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "erf_f32"
  TEST "erf_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "negf_f32"
  TEST "negf_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "reciprocal_f32"
  TEST "reciprocal_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "add_f32"
  TEST "add_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "sub_f32"
  TEST "sub_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "mul_f32"
  TEST "mul_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "div_f32"
  TEST "div_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
  DEFAULT_GEN "tools_check_gen_generators_positive_sequential_vmfb"
)

op_tests(
  NAME "max_f32"
  TEST "max_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "min_f32"
  TEST "min_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
)

op_tests(
  NAME "powf_f32"
  TEST "powf_f32.mlir"
  INSTANCES
    "(4,8)(4,8)"
    "(120,256)(120,256)"
    "(300,450)(300,450)"
  DEFAULT_GEN "tools_check_gen_generators_positive_sequential_vmfb"
)

op_tests(
  NAME "select_f32"
  TEST "select_f32.mlir"
  INSTANCES
    "(4,8)(4,8)(4,8)"
    "(120,256)(120,256)(120,256)"
    "(300,450)(300,450)(300,450)"
  LABELS
    "manual"
)

op_tests(
  NAME "contract_matmul_f32"
  TEST "contract_matmul_f32.mlir"
  INSTANCES
    "(4,8)(8,4)"
    "(120,256)(256,300)"
    "(300,100)(100,450)"
)

op_tests(
  NAME "batch_reduce_matmul_f32"
  TEST "batch_reduce_matmul_f32.mlir"
  INSTANCES
    "(2,4,8)(2,8,4)"
    "(3,12,25)(3,25,30)"
)

op_tests(
  NAME "batch_mmt4d_f32"
  TEST "batch_mmt4d_f32.mlir"
  INSTANCES
    "(2,1,2,4,4)(2,2,2,4,4)"
    "(3,2,3,4,2)(3,3,3,2,2), [manual]"
)

op_tests(
  NAME "matvec_f32"
  TEST "matvec_f32.mlir"
  INSTANCES
    "(4,8)(8)"
    "(120,256)(256)"
    "(300,450)(450)"
)

op_tests(
  NAME "vecmat_f32"
  TEST "vecmat_f32.mlir"
  INSTANCES
    "(8)(8,4)"
    "(256)(256,120)"
    "(450)(450,300)"
)

op_tests(
  NAME "batch_matvec_f32"
  TEST "batch_matvec_f32.mlir"
  INSTANCES
    "(2,4,8)(2,8)"
    "(3,120,256)(3,256)"
)

op_tests(
  NAME "batch_vecmat_f32"
  TEST "batch_vecmat_f32.mlir"
  INSTANCES
    "(2,8)(2,8,4)"
    "(3,256)(3,256,120)"
)

op_tests(
  NAME "dot_f32"
  TEST "dot_f32.mlir"
  INSTANCES
    "(8)(8)"
    "(256)(256)"
    "(450)(450)"
)

op_tests(
  NAME "conv_1d_f32"
  TEST "conv_1d_f32.mlir"
  INSTANCES
    "(11)(3)"
    "(258)(5)"
    "(453)(4), [manual]"
)

op_tests(
  NAME "conv_1d_nwc_wcf_f32"
  TEST "conv_1d_nwc_wcf_f32.mlir"
  INSTANCES
    "(1,11,4)(3,4,8)"
    "(1,256,8)(5,8,16)"
)

op_tests(
  NAME "conv_1d_ncw_fcw_f32"
  TEST "conv_1d_ncw_fcw_f32.mlir"
  INSTANCES
    "(1,4,11)(8,4,3)"
    "(1,8,256)(16,8,5)"
)

op_tests(
  NAME "conv_2d_f32"
  TEST "conv_2d_f32.mlir"
  INSTANCES
    "(11,11)(3,3)"
    "(125,256)(5,7), [manual]"
)

op_tests(
  NAME "conv_3d_f32"
  TEST "conv_3d_f32.mlir"
  INSTANCES
    "(7,7,7)(3,3,3)"
    "(12,15,20)(2,3,4)"
)

op_tests(
  NAME "conv_3d_ndhwc_dhwcf_f32"
  TEST "conv_3d_ndhwc_dhwcf_f32.mlir"
  INSTANCES
    "(1,4,4,4,2)(2,2,2,2,4)"
    "(1,6,6,6,2)(3,3,3,2,4)"
)

op_tests(
  NAME "conv_3d_ncdhw_fcdhw_f32"
  TEST "conv_3d_ncdhw_fcdhw_f32.mlir"
  INSTANCES
    "(1,2,4,4,4)(4,2,2,2,2)"
    "(1,2,6,6,6)(4,2,3,3,3)"
)

op_tests(
  NAME "pooling_nhwc_sum_f32"
  TEST "pooling_nhwc_sum_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_sum_stride2_f32"
  TEST "pooling_nhwc_sum_stride2_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_sum_dilation2_f32"
  TEST "pooling_nhwc_sum_dilation2_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_f32"
  TEST "pooling_nhwc_max_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_stride2_f32"
  TEST "pooling_nhwc_max_stride2_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_max_dilation2_f32"
  TEST "pooling_nhwc_max_dilation2_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "pooling_nhwc_min_f32"
  TEST "pooling_nhwc_min_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3)"
    "(1,120,256,8)(5,7)"
)

op_tests(
  NAME "depthwise_conv_1d_nwc_wc_f32"
  TEST "depthwise_conv_1d_nwc_wc_f32.mlir"
  INSTANCES
    "(1,11,4)(3,4)"
    "(1,256,8)(5,8)"
)

op_tests(
  NAME "depthwise_conv_1d_ncw_cw_f32"
  TEST "depthwise_conv_1d_ncw_cw_f32.mlir"
  INSTANCES
    "(1,4,11)(4,3)"
    "(1,8,256)(8,5)"
)

op_tests(
  NAME "depthwise_conv_1d_nwc_wcm_f32"
  TEST "depthwise_conv_1d_nwc_wcm_f32.mlir"
  INSTANCES
    "(1,11,4)(3,4,2)"
    "(1,256,8)(5,8,1)"
)

op_tests(
  NAME "depthwise_conv_2d_nhwc_hwcm_f32"
  TEST "depthwise_conv_2d_nhwc_hwcm_f32.mlir"
  INSTANCES
    "(1,11,11,4)(3,3,4,2)"
    "(1,120,256,8)(5,5,8,1)"
)

op_tests(
  NAME "depthwise_conv_3d_ndhwc_dhwc_f32"
  TEST "depthwise_conv_3d_ndhwc_dhwc_f32.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3,4)"
    "(1,12,15,20,8)(2,3,4,8)"
)

op_tests(
  NAME "depthwise_conv_3d_ndhwc_dhwcm_f32"
  TEST "depthwise_conv_3d_ndhwc_dhwcm_f32.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3,4,2)"
    "(1,12,15,20,8)(2,3,4,8,1)"
)

op_tests(
  NAME "depthwise_conv_3d_ncdhw_cdhw_f32"
  TEST "depthwise_conv_3d_ncdhw_cdhw_f32.mlir"
  INSTANCES
    "(1,4,7,7,7)(4,3,3,3)"
    "(1,8,12,15,20)(8,2,3,4)"
)

op_tests(
  NAME "conv_2d_nchw_fchw_f32"
  TEST "conv_2d_nchw_fchw_f32.mlir"
  INSTANCES
    "(1,4,6,6)(4,4,3,3)"
    "(1,8,8,8)(16,8,3,3)"
)

op_tests(
  NAME "conv_2d_nhwc_fhwc_f32"
  TEST "conv_2d_nhwc_fhwc_f32.mlir"
  INSTANCES
    "(1,6,6,4)(4,3,3,4)"
    "(1,8,8,8)(16,3,3,8)"
)

op_tests(
  NAME "conv_2d_nhwgc_gfhwc_f32"
  TEST "conv_2d_nhwgc_gfhwc_f32.mlir"
  INSTANCES
    "(1,6,6,2,2)(2,4,3,3,2)"
    "(1,8,8,4,2)(4,8,3,3,2)"
)

op_tests(
  NAME "conv_2d_ngchw_gfchw_f32"
  TEST "conv_2d_ngchw_gfchw_f32.mlir"
  INSTANCES
    "(1,2,2,6,6)(2,4,2,3,3)"
    "(1,4,2,8,8)(4,8,2,3,3)"
)

op_tests(
  NAME "conv_2d_ngchw_fgchw_f32"
  TEST "conv_2d_ngchw_fgchw_f32.mlir"
  INSTANCES
    "(1,2,2,6,6)(4,2,2,3,3)"
    "(1,4,2,8,8)(8,4,2,3,3)"
)

op_tests(
  NAME "pooling_nchw_sum_f32"
  TEST "pooling_nchw_sum_f32.mlir"
  INSTANCES
    "(1,4,11,11)(3,3)"
    "(1,8,120,256)(5,7)"
)

op_tests(
  NAME "pooling_nchw_max_f32"
  TEST "pooling_nchw_max_f32.mlir"
  INSTANCES
    "(1,4,11,11)(3,3)"
    "(1,8,120,256)(5,7)"
)

op_tests(
  NAME "pooling_nwc_sum_f32"
  TEST "pooling_nwc_sum_f32.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_nwc_max_f32"
  TEST "pooling_nwc_max_f32.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_nwc_min_f32"
  TEST "pooling_nwc_min_f32.mlir"
  INSTANCES
    "(1,11,4)(3)"
    "(1,120,8)(5)"
)

op_tests(
  NAME "pooling_ncw_sum_f32"
  TEST "pooling_ncw_sum_f32.mlir"
  INSTANCES
    "(1,4,11)(3)"
    "(1,8,120)(5)"
)

op_tests(
  NAME "pooling_ncw_max_f32"
  TEST "pooling_ncw_max_f32.mlir"
  INSTANCES
    "(1,4,11)(3)"
    "(1,8,120)(5)"
)

op_tests(
  NAME "pooling_ndhwc_sum_f32"
  TEST "pooling_ndhwc_sum_f32.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3)"
    "(1,12,15,20,8)(2,3,4)"
)

op_tests(
  NAME "pooling_ndhwc_max_f32"
  TEST "pooling_ndhwc_max_f32.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3)"
    "(1,12,15,20,8)(2,3,4)"
)

op_tests(
  NAME "pooling_ndhwc_min_f32"
  TEST "pooling_ndhwc_min_f32.mlir"
  INSTANCES
    "(1,7,7,7,4)(3,3,3)"
    "(1,12,15,20,8)(2,3,4)"
)

op_tests(
  NAME "copy_f32"
  TEST "copy_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "fill_rng_2d_f32"
  TEST "fill_rng_2d_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

op_tests(
  NAME "index_f32"
  TEST "index_f32.mlir"
  INSTANCES
    "(4,8)"
    "(120,256)"
    "(300,450)"
)

"""Linalg op_tests list for f32 tests."""

load("//tests/models/linalg:defs.bzl", "op_tests")

def op_tests_f32(name, **kwargs):
    """Registers f32 op tests.

    Args:
      name: The name of the test.
      **kwargs: Additional arguments.
    """
    tags = list(kwargs.pop("tags", []))
    if "f32" not in tags:
        tags.append("f32")
    if "ci" not in tags:
        tags.append("ci")
    op_tests(name = name, tags = tags, **kwargs)

def linalg_op_tests_f32(name = "linalg_op_f32_tests"):
    """Registers Linalg f32 op tests.

    Args:
      name: The name of the test suite.
    """
    op_tests_f32(name = "fill_rank1_f32", instances = ["(8)", "(256)", "(450)"], test = "fill_rank1_f32.mlir")
    op_tests_f32(name = "fill_rank2_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "fill_rank2_f32.mlir")
    op_tests_f32(name = "fill_rank3_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "fill_rank3_f32.mlir")
    op_tests_f32(name = "fill_rank4_f32", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "fill_rank4_f32.mlir")
    op_tests_f32(name = "generic_add_rank1_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "generic_add_rank1_f32.mlir")
    op_tests_f32(name = "generic_add_rank2_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "generic_add_rank2_f32.mlir")

    # TODO: Fix compiler bug with multi-output generic ops.
    op_tests_f32(name = "generic_multi_output_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "generic_multi_output_f32.mlir", tags = ["manual"])
    op_tests_f32(name = "generic_add_rank3_f32", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "generic_add_rank3_f32.mlir")
    op_tests_f32(name = "generic_add_rank4_f32", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "generic_add_rank4_f32.mlir")
    op_tests_f32(name = "matmul_f32", instances = ["(4,8)(8,4)", "(120,256)(256,300)", "(300,100)(100,450)"], test = "matmul_f32.mlir")
    op_tests_f32(name = "batch_matmul_f32", instances = ["(2,4,8)(2,8,4)", "(3,12,25)(3,25,30)"], test = "batch_matmul_f32.mlir")
    op_tests_f32(name = "mmt4d_f32", instances = ["(2,3,4,2)(3,3,2,2)", "(1,2,4,4)(2,2,4,4)"], test = "mmt4d_f32.mlir")
    op_tests_f32(name = "conv_2d_nhwc_hwcf_f32", instances = ["(1,6,6,4)(3,3,4,4)", "(1,8,8,8)(3,3,8,16)"], test = "conv_2d_nhwc_hwcf_f32.mlir")
    op_tests_f32(name = "conv_2d_nhwc_hwcf_stride2_f32", instances = ["(1,6,6,4)(3,3,4,4)", "(1,8,8,8)(3,3,8,16)"], test = "conv_2d_nhwc_hwcf_stride2_f32.mlir")
    op_tests_f32(name = "conv_2d_nhwc_hwcf_dilation2_f32", instances = ["(1,6,6,4)(3,3,4,4)", "(1,8,8,8)(3,3,8,16)"], test = "conv_2d_nhwc_hwcf_dilation2_f32.mlir")
    op_tests_f32(name = "depthwise_conv_2d_nhwc_hwc_f32", instances = ["(1,6,6,4)(3,3,4)", "(1,8,8,8)(3,3,8)"], test = "depthwise_conv_2d_nhwc_hwc_f32.mlir")
    op_tests_f32(name = "depthwise_conv_2d_nhwc_hwc_stride2_f32", instances = ["(1,6,6,4)(3,3,4)", "(1,8,8,8)(3,3,8)"], test = "depthwise_conv_2d_nhwc_hwc_stride2_f32.mlir")
    op_tests_f32(name = "depthwise_conv_2d_nhwc_hwc_dilation2_f32", instances = ["(1,6,6,4)(3,3,4)", "(1,8,8,8)(3,3,8)"], test = "depthwise_conv_2d_nhwc_hwc_dilation2_f32.mlir")
    op_tests_f32(name = "depthwise_conv_2d_nchw_chw_f32", instances = ["(1,4,6,6)(4,3,3)", "(1,8,8,8)(8,3,3)"], test = "depthwise_conv_2d_nchw_chw_f32.mlir")
    op_tests_f32(name = "generic_reduction_2d_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "generic_reduction_2d_f32.mlir")
    op_tests_f32(name = "pack_f32", instances = ["(8,16)", "(16,32)", "(128,256)"], test = "pack_f32.mlir")
    op_tests_f32(name = "pack_perm_f32", instances = ["(8,16)", "(16,32)", "(128,256)"], test = "pack_perm_f32.mlir")
    op_tests_f32(name = "pack_padding_f32", instances = ["(7,15)", "(17,33)", "(120,250)"], test = "pack_padding_f32.mlir")
    op_tests_f32(name = "unpack_f32", instances = ["(1,1,8,16)", "(2,2,8,16)", "(16,16,8,16)"], test = "unpack_f32.mlir")
    op_tests_f32(name = "unpack_perm_f32", instances = ["(1,1,8,16)", "(2,2,8,16)", "(3,2,8,16)"], test = "unpack_perm_f32.mlir")
    op_tests_f32(name = "map_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "map_f32.mlir")
    op_tests_f32(name = "reduce_2d_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_2d_f32.mlir")
    op_tests_f32(name = "reduce_2d_dim0_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_2d_dim0_f32.mlir")
    op_tests_f32(name = "reduce_3d_dim1_2_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "reduce_3d_dim1_2_f32.mlir")
    op_tests_f32(name = "reduce_2d_max_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_2d_max_f32.mlir")
    op_tests_f32(name = "transpose_2d_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "transpose_2d_f32.mlir")
    op_tests_f32(name = "transpose_3d_f32", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "transpose_3d_f32.mlir")
    op_tests_f32(name = "broadcast_f32", instances = ["(8)(4,8)", "(256)(120,256)", "(450)(300,450)"], test = "broadcast_f32.mlir")
    op_tests_f32(name = "broadcast_dim1_f32", instances = ["(8)(8,4)", "(450)(450,300)"], test = "broadcast_dim1_f32.mlir")

    # TODO: Fix compiler bug triggered by this instance.
    op_tests_f32(name = "broadcast_dim1_f32_bug", instances = ["(256)(256,120)"], test = "broadcast_dim1_f32.mlir", tags = ["manual"])
    op_tests_f32(name = "elementwise_add_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "elementwise_add_f32.mlir")
    op_tests_f32(name = "abs_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "abs_f32.mlir")
    op_tests_f32(name = "exp_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "exp_f32.mlir")
    op_tests_f32(name = "log_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "log_f32.mlir")
    op_tests_f32(name = "ceil_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "ceil_f32.mlir")
    op_tests_f32(name = "floor_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "floor_f32.mlir")
    op_tests_f32(name = "round_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "round_f32.mlir")
    op_tests_f32(name = "sqrt_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "sqrt_f32.mlir")
    op_tests_f32(name = "rsqrt_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "rsqrt_f32.mlir")
    op_tests_f32(name = "square_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "square_f32.mlir")
    op_tests_f32(name = "tanh_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "tanh_f32.mlir")
    op_tests_f32(name = "erf_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "erf_f32.mlir")
    op_tests_f32(name = "negf_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "negf_f32.mlir")
    op_tests_f32(name = "reciprocal_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reciprocal_f32.mlir")
    op_tests_f32(name = "add_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "add_f32.mlir")
    op_tests_f32(name = "sub_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "sub_f32.mlir")
    op_tests_f32(name = "mul_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "mul_f32.mlir")
    op_tests_f32(name = "div_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "div_f32.mlir", default_gen = "//tools/check_gen/generators:positive_sequential_vmfb")
    op_tests_f32(name = "max_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "max_f32.mlir")
    op_tests_f32(name = "min_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "min_f32.mlir")
    op_tests_f32(name = "powf_f32", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "powf_f32.mlir", default_gen = "//tools/check_gen/generators:positive_sequential_vmfb")
    op_tests_f32(name = "select_f32", instances = ["(4,8)(4,8)(4,8)", "(120,256)(120,256)(120,256)", "(300,450)(300,450)(300,450)"], test = "select_f32.mlir", tags = ["manual"])
    op_tests_f32(name = "contract_matmul_f32", instances = ["(4,8)(8,4)", "(120,256)(256,300)", "(300,100)(100,450)"], test = "contract_matmul_f32.mlir")
    op_tests_f32(name = "batch_reduce_matmul_f32", instances = ["(2,4,8)(2,8,4)", "(3,12,25)(3,25,30)"], test = "batch_reduce_matmul_f32.mlir")
    op_tests_f32(name = "batch_mmt4d_f32", instances = ["(2,1,2,4,4)(2,2,2,4,4)", "(3,2,3,4,2)(3,3,3,2,2)"], test = "batch_mmt4d_f32.mlir")
    op_tests_f32(name = "matvec_f32", instances = ["(4,8)(8)", "(120,256)(256)", "(300,450)(450)"], test = "matvec_f32.mlir")
    op_tests_f32(name = "vecmat_f32", instances = ["(8)(8,4)", "(256)(256,120)", "(450)(450,300)"], test = "vecmat_f32.mlir")
    op_tests_f32(name = "batch_matvec_f32", instances = ["(2,4,8)(2,8)", "(3,120,256)(3,256)"], test = "batch_matvec_f32.mlir")
    op_tests_f32(name = "batch_vecmat_f32", instances = ["(2,8)(2,8,4)", "(3,256)(3,256,120)"], test = "batch_vecmat_f32.mlir")
    op_tests_f32(name = "dot_f32", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "dot_f32.mlir")
    op_tests_f32(name = "conv_1d_f32", instances = ["(11)(3)", "(258)(5)", "(453)(4)"], test = "conv_1d_f32.mlir")
    op_tests_f32(name = "conv_1d_nwc_wcf_f32", instances = ["(1,11,4)(3,4,8)", "(1,256,8)(5,8,16)"], test = "conv_1d_nwc_wcf_f32.mlir")
    op_tests_f32(name = "conv_1d_ncw_fcw_f32", instances = ["(1,4,11)(8,4,3)", "(1,8,256)(16,8,5)"], test = "conv_1d_ncw_fcw_f32.mlir")
    op_tests_f32(name = "conv_2d_f32", instances = ["(11,11)(3,3)", "(125,256)(5,7)"], test = "conv_2d_f32.mlir")
    op_tests_f32(name = "conv_3d_f32", instances = ["(7,7,7)(3,3,3)", "(12,15,20)(2,3,4)"], test = "conv_3d_f32.mlir")
    op_tests_f32(name = "conv_3d_ndhwc_dhwcf_f32", instances = ["(1,4,4,4,2)(2,2,2,2,4)", "(1,6,6,6,2)(3,3,3,2,4)"], test = "conv_3d_ndhwc_dhwcf_f32.mlir")
    op_tests_f32(name = "conv_3d_ncdhw_fcdhw_f32", instances = ["(1,2,4,4,4)(4,2,2,2,2)", "(1,2,6,6,6)(4,2,3,3,3)"], test = "conv_3d_ncdhw_fcdhw_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_sum_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_sum_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_sum_stride2_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_sum_stride2_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_sum_dilation2_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_sum_dilation2_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_max_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_max_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_max_stride2_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_max_stride2_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_max_dilation2_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_max_dilation2_f32.mlir")
    op_tests_f32(name = "pooling_nhwc_min_f32", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)"], test = "pooling_nhwc_min_f32.mlir")
    op_tests_f32(name = "depthwise_conv_1d_nwc_wc_f32", instances = ["(1,11,4)(3,4)", "(1,256,8)(5,8)"], test = "depthwise_conv_1d_nwc_wc_f32.mlir")
    op_tests_f32(name = "depthwise_conv_1d_ncw_cw_f32", instances = ["(1,4,11)(4,3)", "(1,8,256)(8,5)"], test = "depthwise_conv_1d_ncw_cw_f32.mlir")
    op_tests_f32(name = "depthwise_conv_1d_nwc_wcm_f32", instances = ["(1,11,4)(3,4,2)", "(1,256,8)(5,8,1)"], test = "depthwise_conv_1d_nwc_wcm_f32.mlir")
    op_tests_f32(name = "depthwise_conv_2d_nhwc_hwcm_f32", instances = ["(1,11,11,4)(3,3,4,2)", "(1,120,256,8)(5,5,8,1)"], test = "depthwise_conv_2d_nhwc_hwcm_f32.mlir")
    op_tests_f32(name = "depthwise_conv_3d_ndhwc_dhwc_f32", instances = ["(1,7,7,7,4)(3,3,3,4)", "(1,12,15,20,8)(2,3,4,8)"], test = "depthwise_conv_3d_ndhwc_dhwc_f32.mlir")
    op_tests_f32(name = "depthwise_conv_3d_ndhwc_dhwcm_f32", instances = ["(1,7,7,7,4)(3,3,3,4,2)", "(1,12,15,20,8)(2,3,4,8,1)"], test = "depthwise_conv_3d_ndhwc_dhwcm_f32.mlir")
    op_tests_f32(name = "depthwise_conv_3d_ncdhw_cdhw_f32", instances = ["(1,4,7,7,7)(4,3,3,3)", "(1,8,12,15,20)(8,2,3,4)"], test = "depthwise_conv_3d_ncdhw_cdhw_f32.mlir")
    op_tests_f32(name = "conv_2d_nchw_fchw_f32", instances = ["(1,4,6,6)(4,4,3,3)", "(1,8,8,8)(16,8,3,3)"], test = "conv_2d_nchw_fchw_f32.mlir")
    op_tests_f32(name = "conv_2d_nhwc_fhwc_f32", instances = ["(1,6,6,4)(4,3,3,4)", "(1,8,8,8)(16,3,3,8)"], test = "conv_2d_nhwc_fhwc_f32.mlir")
    op_tests_f32(name = "conv_2d_nhwgc_gfhwc_f32", instances = ["(1,6,6,2,2)(2,4,3,3,2)", "(1,8,8,4,2)(4,8,3,3,2)"], test = "conv_2d_nhwgc_gfhwc_f32.mlir")
    op_tests_f32(name = "conv_2d_ngchw_gfchw_f32", instances = ["(1,2,2,6,6)(2,4,2,3,3)", "(1,4,2,8,8)(4,8,2,3,3)"], test = "conv_2d_ngchw_gfchw_f32.mlir")
    op_tests_f32(name = "conv_2d_ngchw_fgchw_f32", instances = ["(1,2,2,6,6)(4,2,2,3,3)", "(1,4,2,8,8)(8,4,2,3,3)"], test = "conv_2d_ngchw_fgchw_f32.mlir")
    op_tests_f32(name = "pooling_nchw_sum_f32", instances = ["(1,4,11,11)(3,3)", "(1,8,120,256)(5,7)"], test = "pooling_nchw_sum_f32.mlir")
    op_tests_f32(name = "pooling_nchw_max_f32", instances = ["(1,4,11,11)(3,3)", "(1,8,120,256)(5,7)"], test = "pooling_nchw_max_f32.mlir")
    op_tests_f32(name = "pooling_nwc_sum_f32", instances = ["(1,11,4)(3)", "(1,120,8)(5)"], test = "pooling_nwc_sum_f32.mlir")
    op_tests_f32(name = "pooling_nwc_max_f32", instances = ["(1,11,4)(3)", "(1,120,8)(5)"], test = "pooling_nwc_max_f32.mlir")
    op_tests_f32(name = "pooling_nwc_min_f32", instances = ["(1,11,4)(3)", "(1,120,8)(5)"], test = "pooling_nwc_min_f32.mlir")
    op_tests_f32(name = "pooling_ncw_sum_f32", instances = ["(1,4,11)(3)", "(1,8,120)(5)"], test = "pooling_ncw_sum_f32.mlir")
    op_tests_f32(name = "pooling_ncw_max_f32", instances = ["(1,4,11)(3)", "(1,8,120)(5)"], test = "pooling_ncw_max_f32.mlir")
    op_tests_f32(name = "pooling_ndhwc_sum_f32", instances = ["(1,7,7,7,4)(3,3,3)", "(1,12,15,20,8)(2,3,4)"], test = "pooling_ndhwc_sum_f32.mlir")
    op_tests_f32(name = "pooling_ndhwc_max_f32", instances = ["(1,7,7,7,4)(3,3,3)", "(1,12,15,20,8)(2,3,4)"], test = "pooling_ndhwc_max_f32.mlir")
    op_tests_f32(name = "pooling_ndhwc_min_f32", instances = ["(1,7,7,7,4)(3,3,3)", "(1,12,15,20,8)(2,3,4)"], test = "pooling_ndhwc_min_f32.mlir")

    # Softmax fails to compile when targeting only NPU (Issue 1).
    # 'vector.store' op write affecting operations on global resources are restricted
    # to workgroup distributed contexts.
    # Tagging as manual to skip for now.
    op_tests(
        name = "softmax_f32",
        instances = ["(4,8)", "(120,256)", "(300,450)"],
        test = "softmax_f32.mlir",
        tags = ["manual", "f32"],
    )
    op_tests_f32(name = "copy_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "copy_f32.mlir")
    op_tests_f32(name = "fill_rng_2d_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "fill_rng_2d_f32.mlir")
    op_tests_f32(name = "index_f32", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "index_f32.mlir")
    # TODO: Enable this test when compiler crash is fixed. Upstream MLIR
    # linalg.winograd_filter_transform crashes in IREE CPU fallback because it
    # lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).
    # op_tests_f32(name = "winograd_filter_transform_f32", instances = ["(4,3,3,8)", "(8,3,3,16)"], test = "winograd_filter_transform_f32.mlir")

    # TODO: Enable this test when compiler crash is fixed. Upstream MLIR
    # linalg.winograd_input_transform crashes in IREE CPU fallback because it
    # lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).
    # op_tests_f32(name = "winograd_input_transform_f32", instances = ["(1,6,6,8)(4,4,2,2,1,8)"], test = "winograd_input_transform_f32.mlir")

    # TODO: Enable this test when compiler crash is fixed. Upstream MLIR
    # linalg.winograd_output_transform crashes in IREE CPU fallback because it
    # lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).
    # op_tests_f32(name = "winograd_output_transform_f32", instances = ["(4,4,2,2,1,8)(1,4,4,8)"], test = "winograd_output_transform_f32.mlir")

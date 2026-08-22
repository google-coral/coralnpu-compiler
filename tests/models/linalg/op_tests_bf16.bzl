"""Linalg op_tests list for bf16 tests."""

load("//tests/models/linalg:defs.bzl", "op_tests")

def linalg_op_tests_bf16(name = "linalg_op_bf16_tests", check_mlir_files = None, generation_targets = None, target_dir = "generated_bf16"):
    """Registers Linalg bf16 op tests.

    Args:
      name: The name of the test suite.
      check_mlir_files: Optional list to collect check MLIR file paths.
      generation_targets: Optional list to collect check_gen filegroup labels.
      target_dir: Subdirectory where static check test files are stored.
    """

    def op_tests_bf16(name, **kwargs):
        tags = list(kwargs.pop("tags", []))
        if "bf16" not in tags:
            tags.append("bf16")
        if "ci" not in tags:
            tags.append("ci")
        op_tests(
            name = name,
            tags = tags,
            check_mlir_files = check_mlir_files,
            generation_targets = generation_targets,
            target_dir = target_dir,
            **kwargs
        )

    op_tests_bf16(name = "fill_rank1_bf16", instances = ["(8)", "(256)", "(450)"], test = "fill_rank1_bf16.mlir")
    op_tests_bf16(name = "fill_rank2_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "fill_rank2_bf16.mlir")
    op_tests_bf16(name = "fill_rank3_bf16", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "fill_rank3_bf16.mlir")
    op_tests_bf16(name = "fill_rank4_bf16", instances = ["(2,2,3,2)", "(2,3,4,50)", "(1,1,5,400)"], test = "fill_rank4_bf16.mlir")
    op_tests_bf16(name = "generic_add_rank1_bf16", instances = ["(8)(8)", "(256)(256)", "(450)(450)"], test = "generic_add_rank1_bf16.mlir")
    op_tests_bf16(name = "generic_add_rank2_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "generic_add_rank2_bf16.mlir")

    # TODO: Fix compiler bug with multi-output generic ops.
    op_tests_bf16(name = "generic_multi_output_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "generic_multi_output_bf16.mlir", tags = ["manual"])
    op_tests_bf16(name = "generic_add_rank3_bf16", instances = ["(2,3,4)(2,3,4)", "(10,20,30)(10,20,30)", "(5,100,2)(5,100,2)"], test = "generic_add_rank3_bf16.mlir")
    op_tests_bf16(name = "generic_add_rank4_bf16", instances = ["(2,2,3,2)(2,2,3,2)", "(2,3,4,50)(2,3,4,50)", "(1,1,5,400)(1,1,5,400)"], test = "generic_add_rank4_bf16.mlir")
    op_tests_bf16(name = "matmul_bf16", instances = ["(4,8)(8,4)", ("(120,256)(256,300)", ["manual"]), ("(300,100)(100,450)", ["manual"])], test = "matmul_bf16.mlir")
    op_tests_bf16(name = "batch_matmul_bf16", instances = ["(2,4,8)(2,8,4)", ("(3,12,25)(3,25,30)", ["manual"])], test = "batch_matmul_bf16.mlir")
    op_tests_bf16(name = "mmt4d_bf16", instances = [("(1,2,4,4)(2,2,4,4)", ["manual"]), ("(10,12,4,4)(12,12,4,4)", ["manual"]), ("(24,25,4,4)(25,25,4,4)", ["manual"])], test = "mmt4d_bf16.mlir")
    op_tests_bf16(name = "conv_2d_nhwc_hwcf_bf16", instances = ["(1,6,6,4)(3,3,4,4)", "(1,8,8,8)(3,3,8,16)", "(1,65,67,33)(3,3,33,17)"], test = "conv_2d_nhwc_hwcf_bf16.mlir")
    op_tests_bf16(name = "conv_2d_nhwc_hwcf_stride2_bf16", instances = ["(1,6,6,4)(3,3,4,4)", "(1,8,8,8)(3,3,8,16)", "(1,65,67,33)(3,3,33,17)"], test = "conv_2d_nhwc_hwcf_stride2_bf16.mlir")
    op_tests_bf16(name = "conv_2d_nhwc_hwcf_dilation2_bf16", instances = ["(1,6,6,4)(3,3,4,4)", "(1,8,8,8)(3,3,8,16)", "(1,65,67,33)(3,3,33,17)"], test = "conv_2d_nhwc_hwcf_dilation2_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_2d_nhwc_hwc_bf16", instances = ["(1,6,6,4)(3,3,4)", "(1,8,8,8)(3,3,8)", "(1,65,67,33)(3,3,33)"], test = "depthwise_conv_2d_nhwc_hwc_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_2d_nhwc_hwc_stride2_bf16", instances = ["(1,6,6,4)(3,3,4)", "(1,8,8,8)(3,3,8)", "(1,65,67,33)(3,3,33)"], test = "depthwise_conv_2d_nhwc_hwc_stride2_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_2d_nhwc_hwc_dilation2_bf16", instances = ["(1,6,6,4)(3,3,4)", "(1,8,8,8)(3,3,8)", "(1,65,67,33)(3,3,33)"], test = "depthwise_conv_2d_nhwc_hwc_dilation2_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_2d_nchw_chw_bf16", instances = ["(1,4,6,6)(4,3,3)", "(1,8,8,8)(8,3,3)", "(1,33,65,67)(33,3,3)"], test = "depthwise_conv_2d_nchw_chw_bf16.mlir")
    op_tests_bf16(name = "generic_reduction_2d_bf16", instances = ["(4,8)", ("(120,256)", ["manual"]), ("(300,450)", ["manual"])], test = "generic_reduction_2d_bf16.mlir")
    op_tests_bf16(name = "pack_bf16", instances = ["(8,16)", "(16,32)", "(128,256)"], test = "pack_bf16.mlir")
    op_tests_bf16(name = "pack_perm_bf16", instances = ["(8,16)", "(16,32)", "(128,256)"], test = "pack_perm_bf16.mlir")
    op_tests_bf16(name = "pack_padding_bf16", instances = [("(7,15)", ["manual"]), ("(17,33)", ["manual"]), "(120,250)"], test = "pack_padding_bf16.mlir")
    op_tests_bf16(name = "unpack_bf16", instances = ["(1,1,8,16)", "(2,2,8,16)", "(16,16,8,16)"], test = "unpack_bf16.mlir")
    op_tests_bf16(name = "unpack_perm_bf16", instances = ["(1,1,8,16)", "(2,2,8,16)", "(3,2,8,16)"], test = "unpack_perm_bf16.mlir")
    op_tests_bf16(name = "map_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "map_bf16.mlir")
    op_tests_bf16(name = "reduce_2d_bf16", instances = ["(4,8)", ("(120,256)", ["manual"]), ("(300,450)", ["manual"])], test = "reduce_2d_bf16.mlir")
    op_tests_bf16(name = "reduce_2d_dim0_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_2d_dim0_bf16.mlir")
    op_tests_bf16(name = "reduce_3d_dim1_2_bf16", instances = ["(2,3,4)", ("(10,20,30)", ["manual"]), ("(5,100,2)", ["manual"])], test = "reduce_3d_dim1_2_bf16.mlir")
    op_tests_bf16(name = "reduce_2d_max_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reduce_2d_max_bf16.mlir")
    op_tests_bf16(name = "transpose_2d_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "transpose_2d_bf16.mlir")
    op_tests_bf16(name = "transpose_3d_bf16", instances = ["(2,3,4)", "(10,20,30)", "(5,100,2)"], test = "transpose_3d_bf16.mlir")
    op_tests_bf16(name = "broadcast_bf16", instances = ["(8)(4,8)", "(256)(120,256)", "(450)(300,450)"], test = "broadcast_bf16.mlir")
    op_tests_bf16(name = "broadcast_dim1_bf16", instances = ["(8)(8,4)", "(256)(256,120)", "(450)(450,300)"], test = "broadcast_dim1_bf16.mlir")
    op_tests_bf16(name = "elementwise_add_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "elementwise_add_bf16.mlir")
    op_tests_bf16(name = "abs_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "abs_bf16.mlir")
    op_tests_bf16(name = "exp_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "exp_bf16.mlir")
    op_tests_bf16(name = "log_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "log_bf16.mlir")
    op_tests_bf16(name = "ceil_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "ceil_bf16.mlir")
    op_tests_bf16(name = "floor_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "floor_bf16.mlir")
    op_tests_bf16(name = "round_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "round_bf16.mlir")
    op_tests_bf16(name = "sqrt_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "sqrt_bf16.mlir")
    op_tests_bf16(name = "rsqrt_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "rsqrt_bf16.mlir")
    op_tests_bf16(name = "square_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "square_bf16.mlir")
    op_tests_bf16(name = "tanh_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "tanh_bf16.mlir")
    op_tests_bf16(name = "erf_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "erf_bf16.mlir")
    op_tests_bf16(name = "negf_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "negf_bf16.mlir")
    op_tests_bf16(name = "reciprocal_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "reciprocal_bf16.mlir")
    op_tests_bf16(name = "add_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "add_bf16.mlir")
    op_tests_bf16(name = "sub_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "sub_bf16.mlir")
    op_tests_bf16(name = "mul_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "mul_bf16.mlir")
    op_tests_bf16(name = "div_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "div_bf16.mlir", default_gen = "//tools/check_gen/generators:positive_sequential_vmfb")
    op_tests_bf16(name = "max_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "max_bf16.mlir")
    op_tests_bf16(name = "min_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "min_bf16.mlir")
    op_tests_bf16(name = "powf_bf16", instances = ["(4,8)(4,8)", "(120,256)(120,256)", "(300,450)(300,450)"], test = "powf_bf16.mlir", default_gen = "//tools/check_gen/generators:positive_sequential_vmfb")

    # TODO: Enable this test when generator support for i1 tensor operands is added.
    # op_tests_bf16(name = "select_bf16", instances = ["(4,8)(4,8)(4,8)", "(120,256)(120,256)(120,256)", "(300,450)(300,450)(300,450)"], test = "select_bf16.mlir", tags = ["manual"])
    op_tests_bf16(name = "contract_matmul_bf16", instances = ["(4,8)(8,4)", ("(120,256)(256,300)", ["manual"]), ("(300,100)(100,450)", ["manual"])], test = "contract_matmul_bf16.mlir")
    op_tests_bf16(name = "batch_reduce_matmul_bf16", instances = ["(2,4,8)(2,8,4)", ("(3,12,25)(3,25,30)", ["manual"])], test = "batch_reduce_matmul_bf16.mlir")
    op_tests_bf16(name = "batch_mmt4d_bf16", instances = [("(2,1,2,4,4)(2,2,2,4,4)", ["manual"]), ("(2,6,8,4,4)(2,8,8,4,4)", ["manual"]), ("(2,16,20,4,4)(2,18,20,4,4)", ["manual"])], test = "batch_mmt4d_bf16.mlir")
    op_tests_bf16(name = "matvec_bf16", instances = ["(4,8)(8)", ("(120,256)(256)", ["manual"]), ("(300,450)(450)", ["manual"])], test = "matvec_bf16.mlir")
    op_tests_bf16(name = "vecmat_bf16", instances = ["(8)(8,4)", ("(256)(256,120)", ["manual"]), ("(450)(450,300)", ["manual"])], test = "vecmat_bf16.mlir")
    op_tests_bf16(name = "batch_matvec_bf16", instances = ["(2,4,8)(2,8)", ("(3,120,256)(3,256)", ["manual"])], test = "batch_matvec_bf16.mlir")
    op_tests_bf16(name = "batch_vecmat_bf16", instances = ["(2,8)(2,8,4)", ("(3,256)(3,256,120)", ["manual"])], test = "batch_vecmat_bf16.mlir")
    op_tests_bf16(name = "dot_bf16", instances = ["(8)(8)", ("(256)(256)", ["manual"]), ("(450)(450)", ["manual"])], test = "dot_bf16.mlir")
    op_tests_bf16(name = "conv_1d_bf16", instances = ["(11)(3)", ("(258)(5)", ["manual"]), ("(453)(4)", ["manual"]), ("(135001)(5)", ["manual"])], test = "conv_1d_bf16.mlir")
    op_tests_bf16(name = "conv_1d_nwc_wcf_bf16", instances = ["(1,11,4)(3,4,8)", "(1,256,8)(5,8,16)", "(1,4097,33)(5,33,17)"], test = "conv_1d_nwc_wcf_bf16.mlir")
    op_tests_bf16(name = "conv_1d_ncw_fcw_bf16", instances = ["(1,4,11)(8,4,3)", "(1,8,256)(16,8,5)", "(1,33,4097)(17,33,5)"], test = "conv_1d_ncw_fcw_bf16.mlir")
    op_tests_bf16(name = "conv_2d_bf16", instances = ["(11,11)(3,3)", "(125,256)(5,7)", "(301,451)(5,7)"], test = "conv_2d_bf16.mlir")
    op_tests_bf16(name = "conv_3d_bf16", instances = ["(7,7,7)(3,3,3)", "(12,15,20)(2,3,4)", "(33,65,67)(3,3,3)"], test = "conv_3d_bf16.mlir")
    op_tests_bf16(name = "conv_3d_ndhwc_dhwcf_bf16", instances = ["(1,4,4,4,2)(2,2,2,2,4)", "(1,6,6,6,2)(3,3,3,2,4)", "(1,17,33,35,9)(2,3,3,9,7)"], test = "conv_3d_ndhwc_dhwcf_bf16.mlir")
    op_tests_bf16(name = "conv_3d_ncdhw_fcdhw_bf16", instances = ["(1,2,4,4,4)(4,2,2,2,2)", "(1,2,6,6,6)(4,2,3,3,3)", "(1,9,17,33,35)(7,9,2,3,3)"], test = "conv_3d_ncdhw_fcdhw_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_sum_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_sum_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_sum_stride2_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_sum_stride2_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_sum_dilation2_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_sum_dilation2_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_max_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_max_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_max_stride2_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_max_stride2_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_max_dilation2_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_max_dilation2_bf16.mlir")
    op_tests_bf16(name = "pooling_nhwc_min_bf16", instances = ["(1,11,11,4)(3,3)", "(1,120,256,8)(5,7)", "(1,65,67,33)(3,3)"], test = "pooling_nhwc_min_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_1d_nwc_wc_bf16", instances = ["(1,11,4)(3,4)", "(1,256,8)(5,8)", "(1,4097,33)(5,33)"], test = "depthwise_conv_1d_nwc_wc_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_1d_ncw_cw_bf16", instances = ["(1,4,11)(4,3)", "(1,8,256)(8,5)", "(1,33,4097)(33,5)"], test = "depthwise_conv_1d_ncw_cw_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_1d_nwc_wcm_bf16", instances = ["(1,11,4)(3,4,2)", "(1,256,8)(5,8,1)", "(1,4097,33)(5,33,3)"], test = "depthwise_conv_1d_nwc_wcm_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_2d_nhwc_hwcm_bf16", instances = ["(1,11,11,4)(3,3,4,2)", "(1,120,256,8)(5,5,8,1)", "(1,65,67,33)(3,3,33,3)"], test = "depthwise_conv_2d_nhwc_hwcm_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_3d_ndhwc_dhwc_bf16", instances = ["(1,7,7,7,4)(3,3,3,4)", "(1,12,15,20,8)(2,3,4,8)", "(1,17,33,35,9)(2,3,3,9)"], test = "depthwise_conv_3d_ndhwc_dhwc_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_3d_ndhwc_dhwcm_bf16", instances = ["(1,7,7,7,4)(3,3,3,4,2)", "(1,12,15,20,8)(2,3,4,8,1)", "(1,17,33,35,9)(2,3,3,9,3)"], test = "depthwise_conv_3d_ndhwc_dhwcm_bf16.mlir")
    op_tests_bf16(name = "depthwise_conv_3d_ncdhw_cdhw_bf16", instances = ["(1,4,7,7,7)(4,3,3,3)", "(1,8,12,15,20)(8,2,3,4)", "(1,9,17,33,35)(9,2,3,3)"], test = "depthwise_conv_3d_ncdhw_cdhw_bf16.mlir")
    op_tests_bf16(name = "conv_2d_nchw_fchw_bf16", instances = ["(1,4,6,6)(4,4,3,3)", "(1,8,8,8)(16,8,3,3)", "(1,33,65,67)(17,33,3,3)"], test = "conv_2d_nchw_fchw_bf16.mlir")
    op_tests_bf16(name = "conv_2d_nhwc_fhwc_bf16", instances = ["(1,6,6,4)(4,3,3,4)", "(1,8,8,8)(16,3,3,8)", "(1,65,67,33)(17,3,3,33)"], test = "conv_2d_nhwc_fhwc_bf16.mlir")
    op_tests_bf16(name = "conv_2d_nhwgc_gfhwc_bf16", instances = ["(1,6,6,2,2)(2,4,3,3,2)", "(1,8,8,4,2)(4,8,3,3,2)", "(1,65,67,3,11)(3,7,3,3,11)"], test = "conv_2d_nhwgc_gfhwc_bf16.mlir")
    op_tests_bf16(name = "conv_2d_ngchw_gfchw_bf16", instances = ["(1,2,2,6,6)(2,4,2,3,3)", "(1,4,2,8,8)(4,8,2,3,3)", "(1,3,11,65,67)(3,7,11,3,3)"], test = "conv_2d_ngchw_gfchw_bf16.mlir")
    op_tests_bf16(name = "conv_2d_ngchw_fgchw_bf16", instances = ["(1,2,2,6,6)(4,2,2,3,3)", "(1,4,2,8,8)(8,4,2,3,3)", "(1,3,11,65,67)(7,3,11,3,3)"], test = "conv_2d_ngchw_fgchw_bf16.mlir")
    op_tests_bf16(name = "pooling_nchw_sum_bf16", instances = ["(1,4,11,11)(3,3)", "(1,8,120,256)(5,7)", "(1,33,65,67)(3,3)"], test = "pooling_nchw_sum_bf16.mlir")
    op_tests_bf16(name = "pooling_nchw_max_bf16", instances = ["(1,4,11,11)(3,3)", "(1,8,120,256)(5,7)", "(1,33,65,67)(3,3)"], test = "pooling_nchw_max_bf16.mlir")
    op_tests_bf16(name = "pooling_nwc_sum_bf16", instances = ["(1,11,4)(3)", "(1,120,8)(5)", "(1,4097,33)(5)"], test = "pooling_nwc_sum_bf16.mlir")
    op_tests_bf16(name = "pooling_nwc_max_bf16", instances = ["(1,11,4)(3)", "(1,120,8)(5)", "(1,4097,33)(5)"], test = "pooling_nwc_max_bf16.mlir")
    op_tests_bf16(name = "pooling_nwc_min_bf16", instances = ["(1,11,4)(3)", "(1,120,8)(5)", "(1,4097,33)(5)"], test = "pooling_nwc_min_bf16.mlir")
    op_tests_bf16(name = "pooling_ncw_sum_bf16", instances = ["(1,4,11)(3)", "(1,8,120)(5)", "(1,33,4097)(5)"], test = "pooling_ncw_sum_bf16.mlir")
    op_tests_bf16(name = "pooling_ncw_max_bf16", instances = ["(1,4,11)(3)", "(1,8,120)(5)", "(1,33,4097)(5)"], test = "pooling_ncw_max_bf16.mlir")
    op_tests_bf16(name = "pooling_ndhwc_sum_bf16", instances = ["(1,7,7,7,4)(3,3,3)", "(1,12,15,20,8)(2,3,4)", "(1,17,33,35,9)(2,3,3)"], test = "pooling_ndhwc_sum_bf16.mlir")
    op_tests_bf16(name = "pooling_ndhwc_max_bf16", instances = ["(1,7,7,7,4)(3,3,3)", "(1,12,15,20,8)(2,3,4)", "(1,17,33,35,9)(2,3,3)"], test = "pooling_ndhwc_max_bf16.mlir")
    op_tests_bf16(name = "pooling_ndhwc_min_bf16", instances = ["(1,7,7,7,4)(3,3,3)", "(1,12,15,20,8)(2,3,4)", "(1,17,33,35,9)(2,3,3)"], test = "pooling_ndhwc_min_bf16.mlir")

    # Softmax fails in reference compilation targeting llvm-cpu in check_gen
    # ('vector.store' op write affecting operations on global resources).
    # TODO: Enable this test when check_gen reference compilation for softmax is fixed.
    # op_tests_bf16(
    #     name = "softmax_bf16",
    #     instances = ["(4,8)", "(120,256)", "(300,450)"],
    #     test = "softmax_bf16.mlir",
    #     tags = ["manual", "bf16"],
    # )
    op_tests_bf16(name = "copy_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "copy_bf16.mlir")
    op_tests_bf16(name = "fill_rng_2d_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "fill_rng_2d_bf16.mlir")
    op_tests_bf16(name = "index_bf16", instances = ["(4,8)", "(120,256)", "(300,450)"], test = "index_bf16.mlir")
    # TODO: Enable this test when compiler crash is fixed. Upstream MLIR
    # linalg.winograd_filter_transform crashes in IREE CPU fallback because it
    # lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).
    # op_tests_bf16(name = "winograd_filter_transform_bf16", instances = ["(4,3,3,8)", "(8,3,3,16)"], test = "winograd_filter_transform_bf16.mlir")

    # TODO: Enable this test when compiler crash is fixed. Upstream MLIR
    # linalg.winograd_input_transform crashes in IREE CPU fallback because it
    # lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).
    # op_tests_bf16(name = "winograd_input_transform_bf16", instances = ["(1,6,6,8)(4,4,2,2,1,8)"], test = "winograd_input_transform_bf16.mlir")

    # TODO: Enable this test when compiler crash is fixed. Upstream MLIR
    # linalg.winograd_output_transform crashes in IREE CPU fallback because it
    # lacks PartitionableLoopsInterface (which IREE's LinalgExt variant has).
    # op_tests_bf16(name = "winograd_output_transform_bf16", instances = ["(4,4,2,2,1,8)(1,4,4,8)"], test = "winograd_output_transform_bf16.mlir")

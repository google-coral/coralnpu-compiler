# Linalg Operations to Test

This document lists the Linalg operations we plan to test, grouped by similarity and complexity.

## 1. Elementwise and Simple Operations
These operations perform element-by-element transformations or simple initialization.

*   **`linalg.fill`**
    *   *Description*: Fills a tensor with a scalar value.
    *   *Complexity*: Low
*   **`linalg.generic` (Elementwise Unary/Binary)**
    *   *Description*: Generic elementwise operations (e.g., cast, scale, add, multiply).
    *   *Complexity*: Medium (due to generic indexing maps)

## 2. Contraction Operations
Operations that involve matrix multiplication or similar contraction patterns.

*   **`linalg.matmul`**
    *   *Description*: Standard 2D matrix multiplication ($C = A \times B$).
    *   *Complexity*: Medium
*   **`linalg.batch_matmul`**
    *   *Description*: Batched 2D matrix multiplication.
    *   *Complexity*: Medium-High
*   **`linalg.mmt4d`**
    *   *Description*: Matrix-matrix multiplication on 4D tiled tensors. Often used for hardware-accelerated matmul.
    *   *Complexity*: High

## 3. Convolution Operations
Operations representing convolutions, typical in CNNs.

*   **`linalg.conv_2d_nhwc_hwcf`**
    *   *Description*: 2D convolution with NHWC input and HWCF filter format.
    *   *Complexity*: High
*   **`linalg.depthwise_conv_2d_nhwc_hwc`**
    *   *Description*: 2D depthwise convolution.
    *   *Complexity*: High

## 4. Reduction Operations
Operations that reduce dimensions.

*   **`linalg.generic` (Reduction)**
    *   *Description*: Generic reduction operations (e.g., sum reduction along axes).
    *   *Complexity*: Medium-High

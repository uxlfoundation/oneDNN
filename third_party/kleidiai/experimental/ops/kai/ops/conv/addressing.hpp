//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

/* conv kernels share a lot of similarities in how they address input and
 * output tensors. Consequently, this file contains common approaches to
 * preparing these tensor descriptions. Generic (i.e., untyped) methods are
 * contained within the `kai::ops::addressing` namespace, and typed wrappers
 * are provided within an anonymous namespace within `kai::ops::conv`. The
 * variousmethods are described below.
 */

#pragma once

#include <cstddef>
#include <type_traits>

namespace kai {
namespace ops {
namespace addressing {

/* Pointer array
 * -------------
 *
 * Constructs an array of pointers which point to a `array_rows` x `array_cols`
 * chunk of a tensor. The array of pointers will be written into `dest`.
 *
 * `base_ptr` should point at the first VALID element of the chunk of tensor
 * (i.e., if there's one padded row, and one padded column, then `base_ptr`
 * should point at the element which will be at position (1, 1) in the array).
 * `ld_row` and `ld_col` are in bytes, and describe the strides over rows and
 * columns (respectively) of the NHWC-ordered tensor. `pad_buffer` should point
 * at a suitably sized (and initialised) area of memory which can be addressed
 * by elements of the array which represent padding.
 *
 * `pad_top` and `pad_left` describe the padding on the top and left of the
 * array, respectively, and `valid_rows` and `valid_cols` describe the number
 * of rows and columns between the element pointed to by `base_ptr` and the
 * edge of the image (that is `valid_rows` may be greater than `array_rows` and
 * likewise for the columns).
 */
template <typename T>
void fill_pointer_array(
  size_t element_size,
  T *dest, unsigned int array_rows, unsigned int array_cols,
  T base_ptr, size_t ld_row, size_t ld_col,
  T pad_buffer,
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
);

/* Interleaved multi-point pointer array
 * -------------------------------------
 *
 * For each point in a `output_rows` x `output_cols` array, constructs
 * `kernel_rows` x `kernel_cols` array of pointers. The pointers are
 * interleaved thusly:
 *
 *   for ki in kernel_rows:
 *       for kj in kernel_cols:
 *           for oi in output_rows:
 *               for oj in output_cols:
 *                   get pointer for point (oi*stride_rows + ki, oj*stride_cols + kj)
 *
 * Other arguments are as for `fill_pointer_array`.
 *
 * The name reflects that this is the form of addressing mode used by "generic"
 * depthwise kernels.
 */
void fill_pointer_array_generic_kernel(
  size_t element_size,
  const void **dest,
  unsigned int output_rows, unsigned int output_cols,
  unsigned int kernel_rows, unsigned int kernel_cols,
  unsigned int stride_rows, unsigned int stride_cols,
  const void *base_ptr, size_t ld_row, size_t ld_col,
  const void *pad_buffer,
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
);

/* NCHW-patch addressed by row
 * ---------------------------
 *
 * Construct an array of pointers, each of which points at a row of an
 * NCHW-ordered patch of a tensor. Memory addressed by the pointers may be
 * outside of the original tensor, and should therefore not be written to
 * (modifications will be lost).
 *
 * `dest_row_pointers` should point at a `patch_rows` list of pointers; each of
 * which will point at a 1 x `patch_cols` NCHW-ordered sample of the source
 * tensor.
 *
 * `dest_patch` should point to a `element_size * patch_rows * patch_cols` area
 * of memory which can be written to by this function to form samples of the
 * source tensor.
 *
 * `src_ptr` should point at the first VALID element of the chunk of tensor
 * (i.e., if there's one padded row, and one padded column, then `src_ptr`
 * should point at the element which will be at position (1, 1) in the array).
 * `ld_row` and `ld_col` are in bytes, and describe the strides over rows and
 * columns (respectively) of the NHWC-ordered tensor. If `ld_col` ==
 * `element_size` then copies from the source tensor will be elided and source
 * data may be addressed directly.
 *
 * `pad_row` should point to a `patch_cols` array of (appropriately
 * initialised) padding values.
 *
 * Other arguments are as for `fill_pointer_array`.
 */
void fill_nchw_patch_array(
  size_t element_size,
  const void **dest_row_pointers,  // Array of pointers to each row of the patch
  void *dest_patch,  // Pointer to space which can be used to construct the patch
  unsigned int patch_rows, unsigned int patch_cols,  // Patch size
  const void *src_ptr, size_t ld_row, size_t ld_col,  // Source tensor
  const void *pad_row,  // Pointer to a row of padding values
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
);

void fill_patch_array_generic_kernel(
  size_t element_size,
  const void **dest_pointers,  // Pointers: one per output row per kernel point
  void *dest_patch,  // Pointer to space which can be used to construct the patch
  unsigned int output_rows, unsigned int output_cols,
  unsigned int kernel_rows, unsigned int kernel_cols,
  unsigned int stride_rows, unsigned int stride_cols,
  const void *src_ptr, size_t ld_row, size_t ld_col,  // Source tensor
  const void *pad_row,  // Pointer to a row of padding values
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
);

}  // namespace addressing

namespace {

/* Pointer array
 * -------------
 *
 * See `addressing::fill_pointer_array`. No copies are made by this method,
 * memory pointed to by the pointer array is contained within the base tensor
 * and the padding buffer.
 */
template <typename T>
inline void fill_pointer_array(
  T **dest, unsigned int array_rows, unsigned int array_cols,
  T *base_ptr, size_t ld_row, size_t ld_col,
  T *pad_buffer,
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
)
{
  using P = typename std::conditional<std::is_const<T>::value, const char *, char *>::type;
  addressing::fill_pointer_array(
    sizeof(T), reinterpret_cast<P*>(dest), array_rows, array_cols,
    reinterpret_cast<P>(base_ptr), ld_row, ld_col,
    reinterpret_cast<P>(pad_buffer),
    pad_top, valid_rows,
    pad_left, valid_cols
  );
}


/* Interleaved multi-point pointer array
 * -------------------------------------
 *
 * See `addressing::fill_pointer_array_generic_kernel`. No copies are made by
 * this method, memory pointed to by the pointer array is contained within the
 * base tensor and the padding buffer.
 */
template <typename T>
inline void fill_pointer_array_generic_kernel(
  const T **dest,
  unsigned int output_rows, unsigned int output_cols,
  unsigned int kernel_rows, unsigned int kernel_cols,
  unsigned int stride_rows, unsigned int stride_cols,
  const T *base_ptr, size_t ld_row, size_t ld_col,
  const T *pad_buffer,
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
)
{
  addressing::fill_pointer_array_generic_kernel(
    sizeof(T),
    reinterpret_cast<const void **>(dest),
    output_rows, output_cols,
    kernel_rows, kernel_cols,
    stride_rows, stride_cols,
    reinterpret_cast<const void *>(base_ptr), ld_row, ld_col,
    reinterpret_cast<const void *>(pad_buffer),
    pad_top, valid_rows,
    pad_left, valid_cols
  );
}

template <typename T>
inline void fill_nchw_patch_array(
  const T **dest_row_pointers,  // Array of pointers to each row of the patch
  T *dest_patch,  // Pointer to space which can be used to construct the patch
  unsigned int patch_rows, unsigned int patch_cols,  // Patch size
  const T *src_ptr, size_t ld_row, size_t ld_col,  // Source tensor
  const T *pad_row,  // Pointer to a row of padding values
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
)
{
  addressing::fill_nchw_patch_array(
    sizeof(T),
    reinterpret_cast<const void **>(dest_row_pointers),
    reinterpret_cast<void *>(dest_patch),
    patch_rows, patch_cols,
    reinterpret_cast<const void *>(src_ptr), ld_row, ld_col,
    reinterpret_cast<const void *>(pad_row),
    pad_top, valid_rows,
    pad_left, valid_cols
  );
}

template <typename T>
inline void fill_patch_array_generic_kernel(
  const T **dest_pointers,  // Pointers: one per output row per kernel point
  T *dest_patch,  // Pointer to space which can be used to construct the patch
  unsigned int output_rows, unsigned int output_cols,
  unsigned int kernel_rows, unsigned int kernel_cols,
  unsigned int stride_rows, unsigned int stride_cols,
  const T *src_ptr, size_t ld_row, size_t ld_col,  // Source tensor
  const T *pad_row,  // Pointer to a row of padding values
  unsigned int pad_top, unsigned int valid_rows,
  unsigned int pad_left, unsigned int valid_cols
)
{
  addressing::fill_patch_array_generic_kernel(
    sizeof(T),
    reinterpret_cast<const void **>(dest_pointers),
    reinterpret_cast<void *>(dest_patch),
    output_rows, output_cols,
    kernel_rows, kernel_cols,
    stride_rows, stride_cols,
    reinterpret_cast<const void *>(src_ptr), ld_row, ld_col,
    reinterpret_cast<const void *>(pad_row),
    pad_top, valid_rows,
    pad_left, valid_cols
  );
}

}  // namespace {anonymous}
}  // namespace ops
}  // namespace kai

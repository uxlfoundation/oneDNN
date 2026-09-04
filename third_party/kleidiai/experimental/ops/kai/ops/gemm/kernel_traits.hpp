//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai {
namespace ops {

namespace
{
  template <class T>
  constexpr auto is_sme_impl(int)
    -> decltype(T::is_sme(), std::true_type{})
  {
    return std::true_type{};
  }

  template <class>
  constexpr auto is_sme_impl(...) -> std::false_type
  {
    return std::false_type{};
  }
}

template <class T>
struct is_sme
{
  static constexpr auto value = std::is_same<decltype(is_sme_impl<T>(0)),
                                             std::true_type>::value;
};

}  // namespace ops
}  // namespace kai

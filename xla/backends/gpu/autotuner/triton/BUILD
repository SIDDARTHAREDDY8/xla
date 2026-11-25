load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "dot_search_space",
    hdrs = ["dot_search_space.h"],
    tags = ["avoid_dep"],
    deps = ["//xla/backends/gpu/core/autotuner/triton:dot_search_space"],
)

cc_library(
    name = "triton_configs",
    hdrs = ["triton_configs.h"],
    tags = ["avoid_dep"],
    deps = ["//xla/backends/gpu/core/autotuner/triton:triton_configs"],
)

cc_library(
    name = "cost_model_config_optimization",
    hdrs = ["cost_model_config_optimization.h"],
    tags = ["avoid_dep"],
    deps = ["//xla/backends/gpu/core/autotuner/triton:cost_model_config_optimization"],
)

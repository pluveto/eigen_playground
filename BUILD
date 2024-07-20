cc_binary(
    name = "eigen_playground",
    srcs = glob(["*.cpp"]),
    deps = [
        "@eigen//:eigen"
    ],
    linkopts = select({
        "//conditions:default": ["-fsanitize=address"],
    }),
)

#include <iostream>
#include <sstream>
#include <array>
#include <cassert>

#include <Eigen/CXX11/Tensor>

template <typename DerivedA, typename DerivedB>
bool allclose(const Eigen::TensorBase<DerivedA, Eigen::WriteAccessors> &tensorA,
              const Eigen::TensorBase<DerivedB, Eigen::WriteAccessors> &tensorB,
              const typename DerivedA::Scalar &relativeTolerance = Eigen::NumTraits<typename DerivedA::Scalar>::dummy_precision(),
              const typename DerivedA::Scalar &absoluteTolerance = Eigen::NumTraits<typename DerivedA::Scalar>::epsilon())
{
    auto absoluteDifference = (tensorA.eval() - tensorB.eval()).abs();
    auto toleranceBound = (relativeTolerance * tensorB.abs() + absoluteTolerance);
    Eigen::Tensor<bool, 0> isClose = (toleranceBound - absoluteDifference).unaryExpr([](auto x)
                                                                                     { return x >= 0; })
                                         .all();
    return isClose(0);
}

void example_allclose()
{
    Eigen::Tensor<double, 3> tensor(2, 3, 1);
    tensor.setConstant(1.0);
    Eigen::Tensor<double, 3> tensor2(2, 3, 1);
    tensor2.setConstant(1.0 + 1e-16);

    std::cout << "allclose(tensor, tensor2): " << allclose(tensor, tensor2) << std::endl;

    tensor.setConstant(2.0);
    std::cout << "allclose(tensor, tensor2): " << allclose(tensor, tensor2) << std::endl;
}

template <typename T, int Dims>
Eigen::Tensor<T, Dims> ones(const std::array<Eigen::Index, Dims> &dims)
{
    Eigen::Tensor<T, Dims> tensor(dims);
    tensor.setConstant(T(1));
    return tensor;
}

void example_ones()
{
    Eigen::array<Eigen::Index, 2> shape = {2, 3}; // 2x3x4 的张量
    auto tensor = ones<double, 2>(shape);

    std::cout << tensor << std::endl; // 打印张量
}

template <typename T>
std::string stringify_3d_tensor(const Eigen::Tensor<T, 3> &tensor)
{

    std::ostringstream oss;
    auto n_batchs = tensor.dimension(0);
    for (int i = 0; i < n_batchs; ++i)
    {
        oss << "batch " << i << ":\n";
        auto batch = tensor.chip(i, 0);
        oss << batch << "\n";
    }
    return oss.str();
}
void example_sum()
{
    Eigen::Tensor<int, 3> tensor(2, 3, 4);
    int seq = 0;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                tensor(i, j, k) = seq;
                seq++;
            }
        }
    }

    std::cout << "tensor: " << stringify_3d_tensor(tensor) << std::endl;

    std::cout << "tensor.sum(): " << tensor.sum() << std::endl; // 打印张量元素和

    Eigen::Tensor<int, 0> sum = tensor.sum(); // 打印张量元素和

    assert(sum(0) == (2 * 3 * 4) * (2 * 3 * 4 - 1) / 2);
}

int main(int argc, char const *argv[])
{
    // example_allclose();
    // example_ones();
    example_sum();
    return 0;
}

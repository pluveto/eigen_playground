#include <iostream>
#include <Eigen/CXX11/Tensor>

template <typename DerivedA, typename DerivedB>
bool allclose(const Eigen::TensorBase<DerivedA, Eigen::WriteAccessors> &tensorA,
              const Eigen::TensorBase<DerivedB, Eigen::WriteAccessors> &tensorB,
              const typename DerivedA::Scalar &relativeTolerance = Eigen::NumTraits<typename DerivedA::Scalar>::dummy_precision(),
              const typename DerivedA::Scalar &absoluteTolerance = Eigen::NumTraits<typename DerivedA::Scalar>::epsilon())
{
    auto absoluteDifference = (tensorA.eval() - tensorB.eval()).abs();
    auto toleranceBound = (relativeTolerance * tensorB.abs() + absoluteTolerance);
    Eigen::Tensor<bool, 0> isClose = (toleranceBound - absoluteDifference).unaryExpr([](auto x) { return x >= 0; }).all();
    return isClose(0);
}

int main(int argc, char const *argv[])
{
    Eigen::Tensor<double, 3> tensor(2, 3, 1);
    tensor.setConstant(1.0);
    Eigen::Tensor<double, 3> tensor2(2, 3, 1);
    tensor2.setConstant(1.0 + 1e-16);

    std::cout << "allclose(tensor, tensor2): " << allclose(tensor, tensor2) << std::endl;

    tensor.setConstant(2.0);
    std::cout << "allclose(tensor, tensor2): " << allclose(tensor, tensor2) << std::endl;
    return 0;
}

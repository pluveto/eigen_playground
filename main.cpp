#include <iostream>
#include <Eigen/CXX11/Tensor>

int main(int argc, char const *argv[])
{
    Eigen::Tensor<float, 2> tensor(2, 3);
    tensor.setValues({ {1, 2, 3}, {4, 5, 6} });
    std::cout << tensor << std::endl;
    return 0;
}

#include <ModelOrderReduction/component/kernel/QuadraticManifoldProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static void writeMatrix(const std::string& path, const Eigen::MatrixXd& matrix)
{
    std::ofstream stream(path);
    stream << matrix.rows() << " " << matrix.cols() << "\n";
    stream << std::scientific;
    stream.precision(17);
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
        for (Eigen::Index j = 0; j < matrix.cols(); ++j)
            stream << matrix(i, j)
                   << (j + 1 < matrix.cols() ? ' ' : '\n');
}

static Eigen::MatrixXd readMatrix(const std::string& path)
{
    sofa::component::loader::MatrixLoader<Eigen::MatrixXd> loader;
    loader.setFileName(path);
    loader.load();
    Eigen::MatrixXd matrix;
    loader.getMatrix(matrix);
    if (matrix.rows() == 0 || matrix.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return matrix;
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: " << argv[0]
                  << " <bundle> <q> <output> <dq> <force>\n";
        return 2;
    }

    std::filesystem::create_directories(argv[3]);
    sofa::component::kernel::QuadraticManifoldProjector projector;
    projector.loadFromBundle(argv[1]);

    const Eigen::VectorXd q = readMatrix(argv[2]).col(0);
    const Eigen::VectorXd dq = readMatrix(argv[4]).col(0);
    const Eigen::VectorXd force = readMatrix(argv[5]).col(0);
    writeMatrix(std::string(argv[3]) + "/decode.txt", projector.decode(q));
    writeMatrix(std::string(argv[3]) + "/J.txt", projector.J(q));
    writeMatrix(std::string(argv[3]) + "/applyJ.txt", projector.applyJ(q, dq));
    writeMatrix(
        std::string(argv[3]) + "/project_force.txt",
        projector.project_force(q, force));
    return 0;
}

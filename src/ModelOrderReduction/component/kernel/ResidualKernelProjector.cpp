/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                         residual-kernel decoder primitives                  *
******************************************************************************/
#include <ModelOrderReduction/component/kernel/ResidualKernelProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sofa::component::kernel
{

namespace
{

Eigen::MatrixXd load_matrix(const std::string& path)
{
    sofa::component::loader::MatrixLoader<Eigen::MatrixXd> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXd M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to load (missing or empty): " + path);
    return M;
}

struct Params
{
    std::string kernel;
    std::map<std::string, double> values;
};

Params parse_params(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("cannot open residual-kernel params: " + path);

    Params params;
    if (!std::getline(f, params.kernel))
        throw std::runtime_error("empty residual-kernel params: " + path);

    std::string line;
    while (std::getline(f, line))
    {
        std::istringstream ss(line);
        std::string key;
        double value;
        if (ss >> key >> value)
            params.values[key] = value;
    }
    return params;
}

} // namespace

void ResidualKernelProjector::loadFromBundle(const std::string& bundle_dir)
{
    namespace fs = std::filesystem;
    const fs::path root(bundle_dir);

    const MatrixXd meanMat = load_matrix((root / "mean.txt").string());
    const MatrixXd restMat = load_matrix((root / "rest.txt").string());
    if (meanMat.cols() != 1)
        throw std::runtime_error("mean.txt must have one column");
    if (restMat.cols() != 1)
        throw std::runtime_error("rest.txt must have one column");

    m_mean = meanMat.col(0);
    m_rest = restMat.col(0);
    m_primaryBasis = load_matrix((root / "primary_basis.txt").string());
    m_residualBasis = load_matrix((root / "residual_basis.txt").string());
    m_trainQ = load_matrix((root / "train_q.txt").string());
    m_krrWeights = load_matrix((root / "krr_weights.txt").string());

    const Params params = parse_params((root / "params.txt").string());
    if (params.kernel != "rbf")
        throw std::runtime_error("ResidualKernelProjector supports only global rbf");
    auto lengthIt = params.values.find("length_scale");
    auto ridgeIt = params.values.find("ridge");
    if (lengthIt == params.values.end())
        throw std::runtime_error("params.txt missing length_scale");
    if (ridgeIt == params.values.end())
        throw std::runtime_error("params.txt missing ridge");
    m_lengthScale = lengthIt->second;
    m_ridge = ridgeIt->second;

    if (!(m_lengthScale > 0.0))
        throw std::runtime_error("length_scale must be positive");
    if (m_ridge < 0.0)
        throw std::runtime_error("ridge must be non-negative");
    if (m_rest.size() != m_mean.size())
        throw std::runtime_error("rest size != mean size");
    if (m_primaryBasis.rows() != m_mean.size())
        throw std::runtime_error("primary_basis rows != mean size");
    if (m_residualBasis.rows() != m_mean.size())
        throw std::runtime_error("residual_basis rows != mean size");
    if (m_trainQ.cols() != m_primaryBasis.cols())
        throw std::runtime_error("train_q cols != primary mode count");
    if (m_krrWeights.rows() != m_residualBasis.cols())
        throw std::runtime_error("krr_weights rows != residual mode count");
    if (m_krrWeights.cols() != m_trainQ.rows())
        throw std::runtime_error("krr_weights cols != train_q rows");
}

ResidualKernelProjector::VectorXd
ResidualKernelProjector::kernelVector(const Eigen::Ref<const VectorXd>& q) const
{
    if (q.size() != m_primaryBasis.cols())
        throw std::runtime_error("q size mismatch in residual-kernel projector");

    const double inv2ell2 = 1.0 / (2.0 * m_lengthScale * m_lengthScale);
    VectorXd k(m_trainQ.rows());
    for (Eigen::Index i = 0; i < m_trainQ.rows(); ++i)
    {
        const double d2 = (m_trainQ.row(i).transpose() - q).squaredNorm();
        k(i) = std::exp(-d2 * inv2ell2);
    }
    return k;
}

ResidualKernelProjector::VectorXd
ResidualKernelProjector::decode(const Eigen::Ref<const VectorXd>& q) const
{
    const VectorXd k = kernelVector(q);
    return m_mean + m_primaryBasis * q + m_residualBasis * (m_krrWeights * k);
}

ResidualKernelProjector::MatrixXd
ResidualKernelProjector::J(const Eigen::Ref<const VectorXd>& q) const
{
    const VectorXd k = kernelVector(q);
    MatrixXd dk(m_trainQ.rows(), q.size());
    const double invEll2 = 1.0 / (m_lengthScale * m_lengthScale);
    for (Eigen::Index i = 0; i < m_trainQ.rows(); ++i)
        dk.row(i) = (k(i) * invEll2) * (m_trainQ.row(i) - q.transpose());
    return m_primaryBasis + m_residualBasis * (m_krrWeights * dk);
}

ResidualKernelProjector::VectorXd
ResidualKernelProjector::applyJ(const Eigen::Ref<const VectorXd>& q,
                                const Eigen::Ref<const VectorXd>& dq) const
{
    return J(q) * dq;
}

ResidualKernelProjector::VectorXd
ResidualKernelProjector::project_force(const Eigen::Ref<const VectorXd>& q,
                                       const Eigen::Ref<const VectorXd>& f) const
{
    return J(q).transpose() * f;
}

const std::string& ResidualKernelProjector::projectorName()
{
    static const std::string name = "residual-kernel-rbf";
    return name;
}

} // namespace sofa::component::kernel

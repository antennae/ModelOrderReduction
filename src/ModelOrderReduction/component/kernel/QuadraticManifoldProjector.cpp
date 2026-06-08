#include <ModelOrderReduction/component/kernel/QuadraticManifoldProjector.h>
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

Eigen::MatrixXd loadMatrix(const std::string& path)
{
    sofa::component::loader::MatrixLoader<Eigen::MatrixXd> loader;
    loader.setFileName(path);
    loader.load();
    Eigen::MatrixXd matrix;
    loader.getMatrix(matrix);
    if (matrix.rows() == 0 || matrix.cols() == 0)
        throw std::runtime_error("failed to load (missing or empty): " + path);
    return matrix;
}

struct Params
{
    std::string model;
    std::map<std::string, double> values;
};

Params parseParams(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream.is_open())
        throw std::runtime_error("cannot open quadratic-manifold params: " + path);

    Params params;
    if (!std::getline(stream, params.model))
        throw std::runtime_error("empty quadratic-manifold params: " + path);

    std::string line;
    while (std::getline(stream, line))
    {
        std::istringstream parser(line);
        std::string key;
        double value;
        if (parser >> key >> value)
            params.values[key] = value;
    }
    return params;
}

} // namespace

void QuadraticManifoldProjector::loadFromBundle(const std::string& bundleDir)
{
    namespace fs = std::filesystem;
    const fs::path root(bundleDir);

    const MatrixXd meanMatrix = loadMatrix((root / "mean.txt").string());
    const MatrixXd restMatrix = loadMatrix((root / "rest.txt").string());
    if (meanMatrix.cols() != 1 || restMatrix.cols() != 1)
        throw std::runtime_error("mean.txt and rest.txt must have one column");

    m_mean = meanMatrix.col(0);
    m_rest = restMatrix.col(0);
    m_linearBasis = loadMatrix((root / "linear_basis.txt").string());
    m_quadraticBasis = loadMatrix((root / "quadratic_basis.txt").string());

    const Params params = parseParams((root / "params.txt").string());
    if (params.model != "quadratic_manifold")
        throw std::runtime_error(
            "QuadraticManifoldProjector requires quadratic_manifold params");
    const auto modesIt = params.values.find("n_modes");
    const auto ridgeIt = params.values.find("ridge");
    if (modesIt == params.values.end() || ridgeIt == params.values.end())
        throw std::runtime_error("params.txt missing n_modes or ridge");
    m_ridge = ridgeIt->second;

    const Eigen::Index modes = m_linearBasis.cols();
    if (m_rest.size() != m_mean.size())
        throw std::runtime_error("rest size != mean size");
    if (m_linearBasis.rows() != m_mean.size())
        throw std::runtime_error("linear_basis rows != mean size");
    if (m_quadraticBasis.rows() != m_mean.size())
        throw std::runtime_error("quadratic_basis rows != mean size");
    if (m_quadraticBasis.cols() != modes * modes)
        throw std::runtime_error("quadratic_basis cols != n_modes squared");
    if (static_cast<Eigen::Index>(modesIt->second) != modes)
        throw std::runtime_error("params n_modes != linear_basis cols");
    if (m_ridge < 0.0)
        throw std::runtime_error("ridge must be non-negative");
}

void QuadraticManifoldProjector::checkQ(
    const Eigen::Ref<const VectorXd>& q) const
{
    if (q.size() != m_linearBasis.cols())
        throw std::runtime_error("q size mismatch in quadratic-manifold projector");
}

QuadraticManifoldProjector::VectorXd QuadraticManifoldProjector::decode(
    const Eigen::Ref<const VectorXd>& q) const
{
    checkQ(q);
    const Eigen::Index modes = q.size();
    VectorXd features(modes * modes);
    for (Eigen::Index i = 0; i < modes; ++i)
        for (Eigen::Index j = 0; j < modes; ++j)
            features(i * modes + j) = 0.5 * q(i) * q(j);
    return m_mean + m_linearBasis * q + m_quadraticBasis * features;
}

QuadraticManifoldProjector::MatrixXd QuadraticManifoldProjector::J(
    const Eigen::Ref<const VectorXd>& q) const
{
    checkQ(q);
    const Eigen::Index modes = q.size();
    MatrixXd jacobian = m_linearBasis;
    for (Eigen::Index k = 0; k < modes; ++k)
    {
        for (Eigen::Index j = 0; j < modes; ++j)
        {
            jacobian.col(k) +=
                0.5
                * (m_quadraticBasis.col(k * modes + j)
                   + m_quadraticBasis.col(j * modes + k))
                * q(j);
        }
    }
    return jacobian;
}

QuadraticManifoldProjector::VectorXd QuadraticManifoldProjector::applyJ(
    const Eigen::Ref<const VectorXd>& q,
    const Eigen::Ref<const VectorXd>& dq) const
{
    if (dq.size() != m_linearBasis.cols())
        throw std::runtime_error("dq size mismatch in quadratic-manifold projector");
    return J(q) * dq;
}

QuadraticManifoldProjector::VectorXd QuadraticManifoldProjector::project_force(
    const Eigen::Ref<const VectorXd>& q,
    const Eigen::Ref<const VectorXd>& force) const
{
    if (force.size() != m_mean.size())
        throw std::runtime_error(
            "force size mismatch in quadratic-manifold projector");
    return J(q).transpose() * force;
}

const std::string& QuadraticManifoldProjector::projectorName()
{
    static const std::string name = "quadratic-manifold";
    return name;
}

} // namespace sofa::component::kernel

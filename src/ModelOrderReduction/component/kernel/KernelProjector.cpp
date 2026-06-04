/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                         kPCA Galerkin kernel primitives                     *
******************************************************************************/
#include <ModelOrderReduction/component/kernel/KernelProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/helper/logging/Messaging.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sofa {
namespace component {
namespace kernel {

namespace {

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

struct KernelSpec
{
    std::string type;
    std::map<std::string, double> params;
};

KernelSpec parse_kernel_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("cannot open kernel file: " + path);

    KernelSpec spec;
    std::string line;
    if (!std::getline(f, spec.type))
        throw std::runtime_error("kernel file is empty: " + path);
    // strip trailing whitespace/CR
    while (!spec.type.empty() && std::isspace(static_cast<unsigned char>(spec.type.back())))
        spec.type.pop_back();

    while (std::getline(f, line))
    {
        std::istringstream ss(line);
        std::string key;
        double value;
        if (ss >> key >> value)
            spec.params[key] = value;
    }
    return spec;
}

} // anonymous namespace


// ---------------- LinearKernel ----------------

KernelProjector::MatrixXd
LinearKernel::kernel_matrix(const Eigen::Ref<const MatrixXd>& U,
                            const Eigen::Ref<const MatrixXd>& V) const
{
    return U.transpose() * V;
}

KernelProjector::MatrixXd
LinearKernel::grad_u(const Eigen::Ref<const VectorXd>& /*u*/,
                     const Eigen::Ref<const MatrixXd>& V) const
{
    return V;
}

KernelProjector::MatrixXd
LinearKernel::apply_Ginv(const Eigen::Ref<const VectorXd>& /*u*/,
                         const Eigen::Ref<const MatrixXd>& X) const
{
    return X;
}

const std::string& LinearKernel::kernelName() const
{
    static const std::string n = "linear";
    return n;
}


// ---------------- RBFKernel ----------------

RBFKernel::RBFKernel(double sigma) : m_sigma(sigma)
{
    if (!(sigma > 0.0))
        throw std::runtime_error("RBFKernel: sigma must be positive");
}

// Normalize each row i of A by m_scale[i] (z = A ⊘ s). Returns A unchanged
// when s is empty (unscaled kernel).
static KernelProjector::MatrixXd zspace(const Eigen::Ref<const KernelProjector::MatrixXd>& A,
                                        const Eigen::VectorXd& s)
{
    if (s.size() == 0) return A;
    return s.cwiseInverse().asDiagonal() * A;
}

KernelProjector::MatrixXd
RBFKernel::kernel_matrix(const Eigen::Ref<const MatrixXd>& U,
                         const Eigen::Ref<const MatrixXd>& V) const
{
    const MatrixXd Uz = zspace(U, m_scale);
    const MatrixXd Vz = zspace(V, m_scale);
    const Eigen::RowVectorXd sqn_U = Uz.colwise().squaredNorm();
    const Eigen::RowVectorXd sqn_V = Vz.colwise().squaredNorm();
    MatrixXd d2 = Uz.transpose() * Vz;
    d2 *= -2.0;
    d2.colwise() += sqn_U.transpose();
    d2.rowwise() += sqn_V;
    d2 = d2.cwiseMax(0.0);
    const double inv2sig2 = 1.0 / (2.0 * m_sigma * m_sigma);
    return (-d2 * inv2sig2).array().exp().matrix();
}

KernelProjector::MatrixXd
RBFKernel::grad_u(const Eigen::Ref<const VectorXd>& u,
                  const Eigen::Ref<const MatrixXd>& V) const
{
    // ∇_u k = -(u - v_j) ⊘ s² · k / σ²,  k from z-space distances.
    MatrixXd diff = (-V).colwise() + u;                 // (3N, T) physical
    const MatrixXd zdiff = zspace(diff, m_scale);       // (u - v) ⊘ s
    const VectorXd d2 = zdiff.colwise().squaredNorm();  // (T,)
    const double inv2sig2 = 1.0 / (2.0 * m_sigma * m_sigma);
    const VectorXd k = (-d2 * inv2sig2).array().exp();  // (T,)
    const double inv_sig2 = 1.0 / (m_sigma * m_sigma);
    if (m_scale.size() != 0)
    {
        const VectorXd inv_s2 = m_scale.array().square().inverse();
        diff = inv_s2.asDiagonal() * diff;              // (u - v) ⊘ s²
    }
    diff.array().rowwise() *= (-k * inv_sig2).transpose().array();
    return diff;
}

KernelProjector::MatrixXd
RBFKernel::apply_Ginv(const Eigen::Ref<const VectorXd>& /*u*/,
                      const Eigen::Ref<const MatrixXd>& X) const
{
    const double sig2 = m_sigma * m_sigma;
    if (m_scale.size() == 0) return sig2 * X;
    return sig2 * (m_scale.array().square().matrix().asDiagonal() * X);
}

const std::string& RBFKernel::kernelName() const
{
    static const std::string n = "rbf";
    return n;
}


// ---------------- Factory ----------------

std::unique_ptr<KernelProjector> loadKernelProjectorFromBundle(const std::string& bundle_dir)
{
    namespace fs = std::filesystem;

    const fs::path root(bundle_dir);
    const auto X0_path         = (root / "X0.txt").string();
    const auto snapshots_path  = (root / "snapshots.txt").string();
    const auto alpha_path      = (root / "alpha.txt").string();
    const auto kernel_path     = (root / "kernel.txt").string();

    Eigen::MatrixXd X0_mat        = load_matrix(X0_path);         // (3N, 1)
    Eigen::MatrixXd snapshots_mat = load_matrix(snapshots_path);  // (3N, T)
    Eigen::MatrixXd alpha_mat     = load_matrix(alpha_path);      // (m, T)

    if (X0_mat.cols() != 1)
        throw std::runtime_error("X0.txt must have 1 column");

    KernelSpec spec = parse_kernel_file(kernel_path);

    std::unique_ptr<KernelProjector> p;
    if (spec.type == "linear")
    {
        p = std::make_unique<LinearKernel>();
    }
    else if (spec.type == "rbf")
    {
        auto it = spec.params.find("sigma");
        if (it == spec.params.end())
            throw std::runtime_error("RBF bundle missing sigma in kernel.txt");
        p = std::make_unique<RBFKernel>(it->second);
    }
    else
    {
        throw std::runtime_error("unknown kernel type: " + spec.type);
    }

    Eigen::VectorXd X0 = X0_mat.col(0);

    if (snapshots_mat.rows() != X0.size())
        throw std::runtime_error("snapshots rows != X0 length");
    if (alpha_mat.cols() != snapshots_mat.cols())
        throw std::runtime_error("alpha cols != snapshots cols (T mismatch)");

    Eigen::MatrixXd rigid_mat;
    const auto rigid_path = (root / "rigid_modes.txt").string();
    if (fs::exists(rigid_path))
    {
        rigid_mat = load_matrix(rigid_path);
        if (rigid_mat.rows() != X0.size())
            throw std::runtime_error("rigid_modes rows != X0 length");
        if (rigid_mat.cols() < 1 || rigid_mat.cols() > 3)
            throw std::runtime_error("rigid_modes must have 1-3 columns");
    }

    p->setBundleData(std::move(X0), std::move(snapshots_mat),
                     std::move(alpha_mat), std::move(rigid_mat));

    const auto scale_path = (root / "scale.txt").string();
    if (fs::exists(scale_path))
    {
        Eigen::MatrixXd scale_mat = load_matrix(scale_path);  // (3N, 1)
        if (scale_mat.cols() != 1)
            throw std::runtime_error("scale.txt must have 1 column");
        if (scale_mat.rows() != static_cast<Eigen::Index>(p->nbDofs()))
            throw std::runtime_error("scale.txt rows != 3N");
        p->setScale(scale_mat.col(0));
    }

    return p;
}

} // namespace kernel
} // namespace component
} // namespace sofa

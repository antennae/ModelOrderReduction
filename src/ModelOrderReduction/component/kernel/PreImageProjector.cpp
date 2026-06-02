/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                  Explicit kPCA pre-image reconstruction head                 *
******************************************************************************/
#include <ModelOrderReduction/component/kernel/PreImageProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/helper/logging/Messaging.h>

#include <Eigen/Dense>           // SVD / QR / LDLT
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

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

/// Minimal extractor for a flat JSON numeric field (preimage_config.json is
/// {"eta": .., "eta_t": .., "r": .., "max_iter": .., "tol": .., "baseline": ".."}).
double json_get(const std::string& s, const std::string& key, double def)
{
    const std::string tag = "\"" + key + "\"";
    auto pos = s.find(tag);
    if (pos == std::string::npos) return def;
    pos = s.find(':', pos + tag.size());
    if (pos == std::string::npos) return def;
    const char* start = s.c_str() + pos + 1;
    char* end = nullptr;
    const double v = std::strtod(start, &end);
    return (end == start) ? def : v;
}

} // anonymous namespace


PreImageProjector::PreImageProjector(std::unique_ptr<KernelProjector> kernel,
                                     VectorXd massDiag,
                                     double eta, double eta_t,
                                     int r, int maxIter, double tol)
    : m_kernel(std::move(kernel))
    , m_M(std::move(massDiag))
    , m_eta(eta), m_eta_t(eta_t)
    , m_r(r), m_maxIter(maxIter), m_tol(tol)
{
    auto* rbf = dynamic_cast<RBFKernel*>(m_kernel.get());
    if (!rbf)
        throw std::runtime_error("PreImageProjector: kernel must be RBF "
                                 "(pre-image fixed point is RBF-only)");
    m_sigma = rbf->sigma();

    const MatrixXd& D = m_kernel->snapshots();   // (3N, T)
    const MatrixXd& A = m_kernel->alpha();        // (m, T)
    if (m_M.size() != D.rows())
        throw std::runtime_error("PreImageProjector: mass_diagonal length != 3N");

    m_Jinit = D * A.transpose();                  // (3N, m)
    m_mean  = D.rowwise().mean();                 // (3N,)
}

void PreImageProjector::setRegularization(double eta, double eta_t)
{
    if (eta   >= 0.0) m_eta   = eta;
    if (eta_t >= 0.0) m_eta_t = eta_t;
}

PreImageProjector::VectorXd
PreImageProjector::beta(const Eigen::Ref<const VectorXd>& q) const
{
    const Eigen::Index T = m_kernel->snapshots().cols();
    return m_kernel->alpha().transpose() * q
         + VectorXd::Constant(T, 1.0 / static_cast<double>(T));
}

PreImageProjector::VectorXd
PreImageProjector::uInit(const Eigen::Ref<const VectorXd>& q) const
{
    return m_kernel->snapshots() * beta(q);       // D·β(q) = J_init·q + mean
}

PreImageProjector::MatrixXd
PreImageProjector::Jtilde(const Eigen::Ref<const VectorXd>& u) const
{
    const MatrixXd gk = m_kernel->grad_u(u, m_kernel->snapshots());  // (3N, T)
    return gk * m_kernel->alpha().transpose();                        // (3N, m)
}

PreImageProjector::MatrixXd
PreImageProjector::numerator(const Eigen::Ref<const VectorXd>& u) const
{
    MatrixXd N = Jtilde(u);                                            // (3N, m)
    N.noalias() += m_eta * (m_M.asDiagonal() * m_Jinit);
    return N;
}

PreImageProjector::VectorXd
PreImageProjector::grad(const Eigen::Ref<const VectorXd>& u,
                        const Eigen::Ref<const VectorXd>& q,
                        const Eigen::Ref<const VectorXd>& u_init,
                        const Eigen::Ref<const VectorXd>& u_prev) const
{
    const VectorXd b  = beta(q);
    const MatrixXd gk = m_kernel->grad_u(u, m_kernel->snapshots());   // (3N, T)
    VectorXd g = -(gk * b);                                            // kernel term
    g += m_M.cwiseProduct(m_eta * (u - u_init) + m_eta_t * (u - u_prev));
    return g;
}

PreImageProjector::MatrixXd
PreImageProjector::reducedHessian(const Eigen::Ref<const VectorXd>& u,
                                  const Eigen::Ref<const VectorXd>& q,
                                  const Eigen::Ref<const MatrixXd>& W) const
{
    const double s2 = m_sigma * m_sigma;
    const MatrixXd& D = m_kernel->snapshots();                         // (3N, T)
    MatrixXd diff = (-D).colwise() + u;                                // (3N, T)
    const VectorXd d2 = diff.colwise().squaredNorm();                  // (T,)
    const VectorXd kj = (-d2 / (2.0 * s2)).array().exp();              // (T,)
    const VectorXd w  = beta(q).cwiseProduct(kj);                      // (T,)

    const MatrixXd Wd = W.transpose() * diff;                          // (r, T)
    MatrixXd H = (w.sum() / s2) * (W.transpose() * W);
    const MatrixXd Wd_scaled = Wd * (w / s2).asDiagonal();             // (r, T)
    H.noalias() -= (Wd_scaled * Wd.transpose()) / s2;
    H.noalias() += (m_eta + m_eta_t) * (W.transpose() * (m_M.asDiagonal() * W));
    return H;
}

PreImageProjector::VectorXd
PreImageProjector::solve(const Eigen::Ref<const VectorXd>& q,
                         const Eigen::Ref<const VectorXd>& u_init,
                         const Eigen::Ref<const VectorXd>& u_prev) const
{
    const double s2 = m_sigma * m_sigma;
    const MatrixXd& D = m_kernel->snapshots();                         // (3N, T)
    const VectorXd b = beta(q);
    VectorXd u = u_init;
    const VectorXd anchor   = m_M.cwiseProduct(m_eta * u_init + m_eta_t * u_prev);
    const VectorXd diag_reg = (m_eta + m_eta_t) * m_M;

    for (int it = 0; it < m_maxIter; ++it)
    {
        MatrixXd diff = (-D).colwise() + u;                            // (3N, T)
        const VectorXd d2 = diff.colwise().squaredNorm();              // (T,)
        const VectorXd kj = (-d2 / (2.0 * s2)).array().exp();          // (T,)
        const VectorXd w  = b.cwiseProduct(kj);                        // (T,)
        const VectorXd num = (D * w) / s2 + anchor;                    // (3N,)
        const VectorXd den = diag_reg.array() + (w.sum() / s2);        // (3N,)
        const VectorXd u_new = num.cwiseQuotient(den);
        if ((u_new - u).norm() <= m_tol * (1.0 + u.norm()))
            return u_new;
        u = u_new;
    }
    return u;
}

PreImageProjector::MatrixXd
PreImageProjector::localBasisKNN(const Eigen::Ref<const VectorXd>& u0) const
{
    const MatrixXd& D = m_kernel->snapshots();                         // (3N, T)
    const Eigen::Index T = D.cols();
    const int r = std::min<int>(m_r, static_cast<int>(T));

    VectorXd dist(T);
    for (Eigen::Index j = 0; j < T; ++j)
        dist(j) = (D.col(j) - u0).norm();

    std::vector<Eigen::Index> idx(T);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + r, idx.end(),
                      [&](Eigen::Index a, Eigen::Index c) { return dist(a) < dist(c); });

    MatrixXd sub(D.rows(), r);                                         // centered on u0
    for (int j = 0; j < r; ++j)
        sub.col(j) = D.col(idx[j]) - u0;

    Eigen::JacobiSVD<MatrixXd> svd(sub, Eigen::ComputeThinU);
    const VectorXd& s = svd.singularValues();
    const double s0 = (s.size() ? s(0) : 1.0);
    int keep = 0;
    for (Eigen::Index k = 0; k < s.size(); ++k)
        if (s(k) > 1e-10 * s0) ++keep;
    return svd.matrixU().leftCols(keep);                               // (3N, r')
}

PreImageProjector::MatrixXd
PreImageProjector::augmentWithBaseline(const MatrixXd& Wknn) const
{
    MatrixXd aug(Wknn.rows(), Wknn.cols() + m_Jinit.cols());
    aug.leftCols(Wknn.cols())  = Wknn;
    aug.rightCols(m_Jinit.cols()) = m_Jinit;
    // RANK-TRUNCATE via SVD (mirrors preimage.py): [W | J_init] is generally
    // rank-deficient (J_init lies in span(D), largely already in W). Keeping
    // degenerate columns leaves span(W) — and hence the Galerkin Jacobian —
    // dependent on the arbitrary orthonormal completion, which differs between
    // Eigen and NumPy and breaks parity. SVD keeps only the true span.
    Eigen::JacobiSVD<MatrixXd> svd(aug, Eigen::ComputeThinU);
    const VectorXd& s = svd.singularValues();
    const double s0 = (s.size() ? s(0) : 1.0);
    int keep = 0;
    for (Eigen::Index k = 0; k < s.size(); ++k)
        if (s(k) > 1e-10 * s0) ++keep;
    return svd.matrixU().leftCols(keep);
}

PreImageProjector::VectorXd
PreImageProjector::solveReduced(const Eigen::Ref<const VectorXd>& q,
                                const Eigen::Ref<const VectorXd>& u0,
                                const Eigen::Ref<const VectorXd>& u_prev,
                                const Eigen::Ref<const MatrixXd>& W) const
{
    VectorXd a = VectorXd::Zero(W.cols());
    VectorXd u = u0;
    for (int it = 0; it < m_maxIter; ++it)
    {
        u = u0 + W * a;
        const VectorXd g = W.transpose() * grad(u, q, u0, u_prev);     // (r,)
        const MatrixXd H = reducedHessian(u, q, W);                    // (r, r)
        const VectorXd da = H.ldlt().solve(g);
        a -= da;
        if (da.norm() <= m_tol * (1.0 + a.norm()))
            break;
    }
    return u0 + W * a;
}

PreImageProjector::MatrixXd
PreImageProjector::jacobianLocal(const Eigen::Ref<const VectorXd>& q,
                                 const Eigen::Ref<const VectorXd>& u_init,
                                 const Eigen::Ref<const VectorXd>& u_prev) const
{
    const VectorXd u0 = u_init;                       // baseline = F anchor
    const MatrixXd W  = augmentWithBaseline(localBasisKNN(u0));
    const VectorXd u  = solveReduced(q, u0, u_prev, W);
    const MatrixXd N  = numerator(u);                 // J̃ + ηM·J_init
    const MatrixXd WtKW = reducedHessian(u, q, W);    // (r, r)
    return W * WtKW.ldlt().solve(W.transpose() * N);  // (3N, m)
}


// ---------------- Factory ----------------

std::unique_ptr<PreImageProjector>
loadPreImageProjectorFromBundle(const std::string& bundle_dir)
{
    namespace fs = std::filesystem;
    const fs::path root(bundle_dir);

    std::unique_ptr<KernelProjector> kernel = loadKernelProjectorFromBundle(bundle_dir);
    const Eigen::Index n = kernel->nbDofs();

    // mass_diagonal.txt is optional; default to a unit lumped mass.
    Eigen::VectorXd M;
    const auto mass_path = (root / "mass_diagonal.txt").string();
    if (fs::exists(mass_path))
    {
        Eigen::MatrixXd Mm = load_matrix(mass_path);
        if (Mm.rows() != n || Mm.cols() != 1)
            throw std::runtime_error("mass_diagonal.txt must have shape (3N, 1)");
        M = Mm.col(0);
    }
    else
    {
        msg_warning("PreImageProjector") << "mass_diagonal.txt missing in "
            << bundle_dir << "; falling back to unit lumped mass.";
        M = Eigen::VectorXd::Ones(n);
    }

    // preimage_config.json (optional; defaults mirror PREIMAGE_CONFIG_DEFAULTS).
    double eta = 0.5, eta_t = 0.0, tol = 1e-8;
    int r = 30, maxIter = 100;
    const auto cfg_path = (root / "preimage_config.json").string();
    if (fs::exists(cfg_path))
    {
        std::ifstream f(cfg_path);
        std::stringstream buf;
        buf << f.rdbuf();
        const std::string s = buf.str();
        eta     = json_get(s, "eta", eta);
        eta_t   = json_get(s, "eta_t", eta_t);
        r       = static_cast<int>(json_get(s, "r", r));
        maxIter = static_cast<int>(json_get(s, "max_iter", maxIter));
        tol     = json_get(s, "tol", tol);
    }

    return std::make_unique<PreImageProjector>(std::move(kernel), std::move(M),
                                               eta, eta_t, r, maxIter, tol);
}

} // namespace kernel
} // namespace component
} // namespace sofa

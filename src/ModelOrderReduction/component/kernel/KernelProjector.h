/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                         kPCA Galerkin kernel primitives                     *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/config.h>

#include <Eigen/Core>
#include <memory>
#include <string>

namespace sofa {
namespace component {
namespace kernel {

/** Kernel Galerkin projector — pure C++ helper.
 *
 * Exposes the three primitives derived in `mor_kpca.md` §3–§5:
 *
 *   kernel_matrix(U, V) -> (T_U, T_V)   K[i, j] = k(U[:, i], V[:, j])
 *   grad_u(u, V)        -> (3N, T)      column j = ∇_u k(u, V[:, j])
 *   apply_Ginv(u, X)    -> same shape as X,  y = G(u)^{-1} X
 *
 * where G(u) := [∂φ/∂u]^T [∂φ/∂u] is the derivative Gram matrix
 * (Otto-Padovan-Rowley 2023 Theorem 2.7). Closed forms for both kernels come
 * from Otto-Padovan-Rowley 2023 Table 1.
 *
 * **A note on "Gram matrix".** Two objects get called "Gram":
 *   - the **kernel Gram matrix** K[i, j] = k(u_i, u_j) returned by
 *     `kernel_matrix()`;
 *   - the **derivative Gram matrix** G(u) whose inverse is applied by
 *     `apply_Ginv()`.
 * `mor_kpca.md` reserves the symbol G for the second.
 */
class SOFA_MODELORDERREDUCTION_API KernelProjector
{
public:
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    virtual ~KernelProjector() = default;

    /// Kernel Gram matrix: K[i, j] = k(U[:, i], V[:, j]).
    virtual MatrixXd kernel_matrix(const Eigen::Ref<const MatrixXd>& U,
                                   const Eigen::Ref<const MatrixXd>& V) const = 0;

    /// ∇_u k(u, V[:, j]) for each column j — shape (3N, T).
    virtual MatrixXd grad_u(const Eigen::Ref<const VectorXd>& u,
                            const Eigen::Ref<const MatrixXd>& V) const = 0;

    /// G(u)^{-1} X — same shape as X. (Accepts VectorXd as a single-column matrix.)
    virtual MatrixXd apply_Ginv(const Eigen::Ref<const VectorXd>& u,
                                const Eigen::Ref<const MatrixXd>& X) const = 0;

    /// J(u) · dq — shape (3N,). Avoids building J(u) explicitly:
    ///   J(u) dq = G^{-1}(u) · grad_u(u, X) · (αᵀ dq).
    VectorXd applyJ(const Eigen::Ref<const VectorXd>& u,
                    const Eigen::Ref<const VectorXd>& dq) const
    {
        const VectorXd alpha_dq = m_alpha.transpose() * dq;       // (T,)
        const MatrixXd Gjac     = grad_u(u, m_snapshots);         // (3N, T)
        const MatrixXd Jtilde_dq = Gjac * alpha_dq;               // (3N, 1)
        return apply_Ginv(u, Jtilde_dq).col(0);                   // (3N,)
    }

    /// J(u)ᵀ · f — shape (m,). Mirrors Python KernelProjector.project_force.
    VectorXd project_force(const Eigen::Ref<const VectorXd>& u,
                           const Eigen::Ref<const VectorXd>& f) const
    {
        const MatrixXd Ginv_f = apply_Ginv(u, f);                 // (3N, 1)
        const MatrixXd Gjac   = grad_u(u, m_snapshots);           // (3N, T)
        const VectorXd Gt_f   = Gjac.transpose() * Ginv_f.col(0); // (T,)
        return m_alpha * Gt_f;                                    // (m,)
    }

    /// J(u) — full (3N, m) decoder Jacobian. Use applyJ / project_force instead
    /// when you only need a matrix-vector product, to avoid the (3N, m) alloc.
    MatrixXd J(const Eigen::Ref<const VectorXd>& u) const
    {
        const MatrixXd Gjac = grad_u(u, m_snapshots);             // (3N, T)
        const MatrixXd Jt   = Gjac * m_alpha.transpose();         // (3N, m)
        return apply_Ginv(u, Jt);                                 // (3N, m)
    }

    /// Loaded data accessors.
    const VectorXd& X0()        const { return m_X0; }
    const MatrixXd& snapshots() const { return m_snapshots; }
    const MatrixXd& alpha()     const { return m_alpha; }
    unsigned nbDofs()           const { return static_cast<unsigned>(m_snapshots.rows()); }
    unsigned nbSnapshots()      const { return static_cast<unsigned>(m_snapshots.cols()); }
    unsigned nbModes()          const { return static_cast<unsigned>(m_alpha.rows()); }

    virtual const std::string& kernelName() const = 0;

    /// Install the loaded bundle data. Called by the bundle-loader factory.
    void setBundleData(VectorXd X0, MatrixXd snapshots, MatrixXd alpha)
    {
        m_X0 = std::move(X0);
        m_snapshots = std::move(snapshots);
        m_alpha = std::move(alpha);
    }

protected:
    KernelProjector() = default;

    VectorXd m_X0;          // (3N,)
    MatrixXd m_snapshots;   // (3N, T)
    MatrixXd m_alpha;       // (m, T)
};


/** Linear kernel k(x, y) = x^T y.  G = I, ∇_u k(u, v) = v. */
class SOFA_MODELORDERREDUCTION_API LinearKernel : public KernelProjector
{
public:
    LinearKernel() = default;

    MatrixXd kernel_matrix(const Eigen::Ref<const MatrixXd>& U,
                           const Eigen::Ref<const MatrixXd>& V) const override;
    MatrixXd grad_u(const Eigen::Ref<const VectorXd>& u,
                    const Eigen::Ref<const MatrixXd>& V) const override;
    MatrixXd apply_Ginv(const Eigen::Ref<const VectorXd>& u,
                        const Eigen::Ref<const MatrixXd>& X) const override;
    const std::string& kernelName() const override;
};


/** Gaussian RBF k(x, y) = exp(-||x - y||² / (2σ²)).  G^{-1} = σ² I. */
class SOFA_MODELORDERREDUCTION_API RBFKernel : public KernelProjector
{
public:
    explicit RBFKernel(double sigma);

    double sigma() const { return m_sigma; }

    MatrixXd kernel_matrix(const Eigen::Ref<const MatrixXd>& U,
                           const Eigen::Ref<const MatrixXd>& V) const override;
    MatrixXd grad_u(const Eigen::Ref<const VectorXd>& u,
                    const Eigen::Ref<const MatrixXd>& V) const override;
    MatrixXd apply_Ginv(const Eigen::Ref<const VectorXd>& u,
                        const Eigen::Ref<const MatrixXd>& X) const override;
    const std::string& kernelName() const override;

private:
    double m_sigma;
};


/// Factory: read a kPCA bundle directory and return the right KernelProjector.
SOFA_MODELORDERREDUCTION_API
std::unique_ptr<KernelProjector> loadKernelProjectorFromBundle(const std::string& bundle_dir);

} // namespace kernel
} // namespace component
} // namespace sofa

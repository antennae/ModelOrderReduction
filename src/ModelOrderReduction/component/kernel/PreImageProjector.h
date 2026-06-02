/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                  Explicit kPCA pre-image reconstruction head                 *
*
*  C++ port of src/kpca/preimage.py (RBF only). See docs/methods/preimage_rom.md.
*
*  The OPR-Galerkin head (KernelProjector) decodes by incremental tangent
*  integration with the metric G(u)⁻¹ and drops geometric stiffness. This head
*  instead:
*    - decodes by an EXACT pre-image solve  u = Ψ(q)  (RBF fixed point), and
*    - supplies the implicit-function-theorem Jacobian
*          J(q) = K_pre⁻¹ ( J̃(u) + η M · J_init )
*      via the tractable local-POD reduction (only an r×r inverse), where the
*      q-dependent linear-POD baseline u_init(q)=J_init·q+mean anchors the solve
*      (J_init = D·αᵀ).
******************************************************************************/
#pragma once

#include <ModelOrderReduction/config.h>
#include <ModelOrderReduction/component/kernel/KernelProjector.h>

#include <Eigen/Core>
#include <memory>
#include <string>

namespace sofa {
namespace component {
namespace kernel {

class SOFA_MODELORDERREDUCTION_API PreImageProjector
{
public:
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    /// kernel must be an RBFKernel (with bundle data already installed).
    PreImageProjector(std::unique_ptr<KernelProjector> kernel,
                      VectorXd massDiag,
                      double eta, double eta_t,
                      int r, int maxIter, double tol);

    // ---- bundle accessors (delegated to the kernel projector) ----
    const VectorXd& X0()        const { return m_kernel->X0(); }
    const MatrixXd& snapshots() const { return m_kernel->snapshots(); }
    const MatrixXd& alpha()     const { return m_kernel->alpha(); }
    unsigned nbDofs()      const { return m_kernel->nbDofs(); }
    unsigned nbSnapshots() const { return m_kernel->nbSnapshots(); }
    unsigned nbModes()     const { return m_kernel->nbModes(); }
    const std::string& kernelName() const { return m_kernel->kernelName(); }

    double eta()    const { return m_eta; }
    double eta_t()  const { return m_eta_t; }
    int    r()      const { return m_r; }
    double sigma()  const { return m_sigma; }
    const VectorXd& massDiag() const { return m_M; }
    const MatrixXd& Jinit()    const { return m_Jinit; }  // (3N, m) = D·αᵀ
    const VectorXd& mean()     const { return m_mean; }   // (3N,) snapshot mean

    /// runtime overrides for the η-sweep (negative ⇒ keep bundle value).
    void setRegularization(double eta, double eta_t);
    void setLocalRank(int r) { if (r > 0) m_r = r; }

    // β_j(q) = (αᵀ q)_j + 1/T
    VectorXd beta(const Eigen::Ref<const VectorXd>& q) const;
    // q-dependent linear-POD baseline u_init(q) = J_init·q + mean = D·β(q)
    VectorXd uInit(const Eigen::Ref<const VectorXd>& q) const;

    // J̃(u) = ∇_u k(u, D) · αᵀ  (3N, m)
    MatrixXd Jtilde(const Eigen::Ref<const VectorXd>& u) const;
    // numerator N(u) = J̃(u) + η M · J_init  (3N, m)
    MatrixXd numerator(const Eigen::Ref<const VectorXd>& u) const;

    // ∇_u Φ (3N) = −Σ_j β_j ∇_u k + M⊙(η(u−u_init) + η_t(u−u_prev))
    VectorXd grad(const Eigen::Ref<const VectorXd>& u,
                  const Eigen::Ref<const VectorXd>& q,
                  const Eigen::Ref<const VectorXd>& u_init,
                  const Eigen::Ref<const VectorXd>& u_prev) const;
    // WᵀK_pre W (r×r) without forming the dense 3N×3N Hessian
    MatrixXd reducedHessian(const Eigen::Ref<const VectorXd>& u,
                            const Eigen::Ref<const VectorXd>& q,
                            const Eigen::Ref<const MatrixXd>& W) const;

    // RBF fixed-point exact decode  u = Ψ(q)
    VectorXd solve(const Eigen::Ref<const VectorXd>& q,
                   const Eigen::Ref<const VectorXd>& u_init,
                   const Eigen::Ref<const VectorXd>& u_prev) const;

    // thin POD of the r nearest snapshots to u0 (centered on u0)
    MatrixXd localBasisKNN(const Eigen::Ref<const VectorXd>& u0) const;
    // reduced Newton solve a* = argmin_a Φ(u0 + W a, q); returns u0 + W a*
    VectorXd solveReduced(const Eigen::Ref<const VectorXd>& q,
                          const Eigen::Ref<const VectorXd>& u0,
                          const Eigen::Ref<const VectorXd>& u_prev,
                          const Eigen::Ref<const MatrixXd>& W) const;
    // local-POD reduced Jacobian J(q) = W(WᵀK_preW)⁻¹(WᵀN), q-dep baseline (3N, m)
    MatrixXd jacobianLocal(const Eigen::Ref<const VectorXd>& q,
                           const Eigen::Ref<const VectorXd>& u_init,
                           const Eigen::Ref<const VectorXd>& u_prev) const;
    // Same Jacobian, but evaluated at a SUPPLIED pre-image u instead of
    // re-solving. The mapping reuses apply()'s decoded u = Ψ(q), dropping the
    // redundant reduced-Newton solve (the per-step cost driver). W is still
    // built from the baseline u_init, exactly as in jacobianLocal.
    MatrixXd jacobianAt(const Eigen::Ref<const VectorXd>& q,
                        const Eigen::Ref<const VectorXd>& u_init,
                        const Eigen::Ref<const VectorXd>& u) const;

private:
    /// W ← orth([W_kNN | J_init]) so the η M·J_init Jacobian term is representable.
    MatrixXd augmentWithBaseline(const MatrixXd& Wknn) const;
    /// W (WᵀK_pre(u) W)⁻¹ (WᵀN(u)) — shared by jacobianLocal / jacobianAt.
    MatrixXd jacobianInBasis(const Eigen::Ref<const VectorXd>& q,
                             const Eigen::Ref<const MatrixXd>& W,
                             const Eigen::Ref<const VectorXd>& u) const;

    std::unique_ptr<KernelProjector> m_kernel;  // RBF, bundle installed
    double   m_sigma = 1.0;
    VectorXd m_M;        // (3N,) lumped diagonal mass
    double   m_eta = 0.0;
    double   m_eta_t = 0.0;
    int      m_r = 30;
    int      m_maxIter = 100;
    double   m_tol = 1e-8;
    MatrixXd m_Jinit;    // (3N, m) = D·αᵀ
    VectorXd m_mean;     // (3N,) snapshot mean
};

/// Factory: load an RBF bundle + preimage_config.json + mass_diagonal.txt.
/// mass_diagonal.txt is optional (falls back to ones with a warning).
SOFA_MODELORDERREDUCTION_API
std::unique_ptr<PreImageProjector>
loadPreImageProjectorFromBundle(const std::string& bundle_dir);

} // namespace kernel
} // namespace component
} // namespace sofa

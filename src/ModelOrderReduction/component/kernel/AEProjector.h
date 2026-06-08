/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                         AE-decoder Galerkin projector                       *
*
* Parallel sibling to KernelProjector. Wraps a trained autoencoder decoder
* g_θ : R^m → R^{3N} loaded from a bundle directory containing
*   decoder.ts.pt   TorchScript decoder (col_std unnormalisation baked in)
*   arch.json       architecture spec (m, 3N, hidden width, activation)
*   X0.txt          (3N, 1) reference state, MatrixLoader format
*   col_std.txt     (3N, 1) per-DOF std normaliser, MatrixLoader format
*   encoder.ts.pt   optional TorchScript encoder for q_0 = encode(0)
*   rigid_modes.txt optional (3N, k) orthonormal translation columns
*
* Runtime primitives mirror src/ae/projector.py:
*   decode(q)              = g_θ(q)        ∈ ℝ^{3N}
*   applyJ(q, dq)          = J(q) · dq     ∈ ℝ^{3N}  (single JVP)
*   project_force(q, f)    = J(q)^T · f    ∈ ℝ^m     (single VJP)
*   J(q)                   = ∂g/∂q         ∈ ℝ^{3N×m}  (analytical chain rule)
*   encode(u)              = f_θ(u)        ∈ ℝ^m
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/DecoderProjector.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace sofa::component::kernel {

class SOFA_MODELORDERREDUCTION_API AEProjector : public DecoderProjector
{
public:
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    AEProjector();
    ~AEProjector();

    /// Load decoder.ts.pt + arch.json + X0.txt + col_std.txt (and optional
    /// encoder.ts.pt + rigid_modes.txt) from `bundle_dir`. Throws on failure.
    void loadFromBundle(const std::string& bundle_dir) override;

    /// u_disp = col_std ⊙ g_θ(q). Shape (3N,).
    VectorXd decode(const Eigen::Ref<const VectorXd>& q) const override;

    /// q_init from u_disp via encoder. Available only when encoder.ts.pt is in
    /// the bundle; throws otherwise. Used at scene init.
    VectorXd encode(const Eigen::Ref<const VectorXd>& u_disp) const override;

    /// J(q) = ∂(col_std ⊙ g_θ)/∂q — full dense (3N, m) Jacobian.
    /// Use applyJ / project_force when you only need a matvec to avoid the
    /// (3N, m) alloc. Stored cache in AEMapping calls this once per step.
    MatrixXd J(const Eigen::Ref<const VectorXd>& q) const override;

    /// J(q) · dq via JVP — shape (3N,).
    VectorXd applyJ(const Eigen::Ref<const VectorXd>& q,
                    const Eigen::Ref<const VectorXd>& dq) const override;

    /// J(q)^T · f via VJP — shape (m,). Mirrors KernelProjector::project_force.
    VectorXd project_force(const Eigen::Ref<const VectorXd>& q,
                           const Eigen::Ref<const VectorXd>& f) const override;

    /// Loaded data accessors — match KernelProjector's surface so AEMapping
    /// can mirror KernelPCAMapping verbatim.
    const VectorXd& X0()         const { return m_X0; }
    const MatrixXd& rigidModes() const override { return m_rigidModes; }  // (3N, k), k=0 if absent
    unsigned nbDofs()  const override { return m_nbDofs; }
    unsigned nbModes() const override { return m_nbModes; }
    unsigned nbRigid() const override { return static_cast<unsigned>(m_rigidModes.cols()); }

    static const std::string& projectorName();   // returns "ae"

private:
    // Architecture cached at load time; weights stored as Eigen for fast J().
    // The decoder is a SiLU MLP: y = W_n · σ(W_{n-1} · σ(... σ(W_1·q + b_1) ...) + b_{n-1}) + b_n
    // where σ is SiLU and the last layer has no activation. The Python decoder's
    // last-layer zero-init guarantees g(0) = 0 ⇒ u(q=0) = X0 + col_std⊙0 = X0.
    std::vector<Eigen::MatrixXd> m_weights;   // W_i, indexed 0..n-1
    std::vector<Eigen::VectorXd> m_biases;    // b_i
    Eigen::VectorXd m_col_std;   // (3N,) per-DOF unnormaliser

    // Optional encoder kept as TorchScript module; only used at init.
    // Held via opaque pointer so this header doesn't include <torch/...>.
    struct EncoderImpl;
    std::unique_ptr<EncoderImpl> m_encoder;

    // Per-step forward cache shared by decode/applyJ/project_force. Keyed on
    // q so the runtime can call the primitives in any order at the same
    // linearisation point and pay a single forward pass. mutable because the
    // primitive accessors are logically const.
    mutable Eigen::VectorXd              m_cache_q;       // last q we ran forward on
    mutable std::vector<Eigen::VectorXd> m_cache_z;       // pre-activations per layer
    mutable bool                         m_cache_valid = false;

    /// Refresh m_cache_z if q differs from m_cache_q. No-op when already current.
    void ensureForwardCache(const Eigen::Ref<const VectorXd>& q) const;

    // Bundle metadata
    VectorXd m_X0;
    MatrixXd m_rigidModes;
    unsigned m_nbDofs = 0;
    unsigned m_nbModes = 0;
    std::string m_activation;  // "silu" (only one supported in v1)
};

} // namespace sofa::component::kernel

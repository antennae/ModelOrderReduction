/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   DecoderProjector — common ABC for the q-keyed reduced decoders            *
*   (AE / residual-kernel / quadratic-manifold). C++ mirror of the Python     *
*   Decoder ABC in src/_decoder_base.py. kPCA's KernelProjector is a state-   *
*   keyed Galerkin integrator with no decode() and deliberately does NOT      *
*   derive from this. See docs/superpowers/specs/                             *
*   2026-06-08-r6-decoder-projector-abc-design.md.                            *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/config.h>

#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <vector>

namespace sofa::component::kernel
{

/** q-keyed reduced decoder: u = decode(q), with J(q) for Galerkin assembly.
 *
 * The pure-virtual core (loadFromBundle / decode / J / nbDofs / nbModes) is
 * what compiler-enforces the C++<->NumPy mirror. applyJ / project_force have
 * default impls (J*dq, J^T f, with fail-loud size checks) that AE overrides
 * with a direct JVP/VJP. encode / rigidModes / nbRigid are optional
 * capabilities with sensible defaults (throw / empty / 0); a method overrides
 * only what it actually supports.
 */
class SOFA_MODELORDERREDUCTION_API DecoderProjector
{
public:
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;

    virtual ~DecoderProjector() = default;

    // --- pure-virtual core ---
    virtual void     loadFromBundle(const std::string& bundleDir) = 0;
    virtual VectorXd decode(const Eigen::Ref<const VectorXd>& q) const = 0;
    virtual MatrixXd J(const Eigen::Ref<const VectorXd>& q) const = 0;
    virtual unsigned nbDofs()  const = 0;
    virtual unsigned nbModes() const = 0;

    // --- default impls; override where a closed form is cheaper ---
    virtual VectorXd applyJ(const Eigen::Ref<const VectorXd>& q,
                            const Eigen::Ref<const VectorXd>& dq) const
    {
        if (static_cast<unsigned>(dq.size()) != nbModes())
            throw std::runtime_error("DecoderProjector::applyJ: dq size != nbModes");
        return J(q) * dq;
    }
    virtual VectorXd project_force(const Eigen::Ref<const VectorXd>& q,
                                   const Eigen::Ref<const VectorXd>& f) const
    {
        if (static_cast<unsigned>(f.size()) != nbDofs())
            throw std::runtime_error("DecoderProjector::project_force: f size != nbDofs");
        return J(q).transpose() * f;
    }

    /// J(q)^T applied to one element's nodal force, scattered (3V) into the full
    /// (3N) vector first. Mirrors Python Decoder.project_element; the canonical
    /// reference for the helper's element-local projectOneElement.
    VectorXd project_element(const Eigen::Ref<const VectorXd>& q,
                             const std::vector<unsigned int>& indexList,
                             const Eigen::Ref<const VectorXd>& contrib) const;

    // --- optional capabilities ---
    /// u_disp -> q (initial latent coords). Not needed by the SOFA runtime, so
    /// it throws by default; override where available (AE: TorchScript encoder;
    /// residual/quadratic: linear-POD projection, not yet ported to C++).
    virtual VectorXd encode(const Eigen::Ref<const VectorXd>& u_disp) const;
    /// Rigid-translation columns (Rigidify). Empty by default.
    virtual const MatrixXd& rigidModes() const;
    virtual unsigned        nbRigid() const { return 0; }
};

} // namespace sofa::component::kernel

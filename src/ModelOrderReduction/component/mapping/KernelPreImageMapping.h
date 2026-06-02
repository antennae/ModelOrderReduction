/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   Explicit pre-image reconstruction head (sibling of KernelPCAMapping).
*
*   Where KernelPCAMapping decodes by INCREMENTAL TANGENT INTEGRATION
*   (u_new = u_old + J(u_old)·Δq, J = σ²J̃ frozen per step, never re-projecting
*   onto the manifold) and drops geometric stiffness, this head:
*
*     apply   : EXACT decode  u = Ψ(q)  via the RBF fixed-point pre-image solve
*               (re-projects onto the manifold every step). out = X0 + Ψ(q).
*     applyJ  : du = J(q)·dq,  J(q) = K_pre⁻¹(J̃(u) + ηM·J_init) via the
*     applyJT : dq = J(q)ᵀ·du   tractable local-POD reduced path (r×r inverse).
*     applyDJT: geometric stiffness K_geo = (∂J/∂q)ᵀf  (Stage B; FD of cached J).
*
*   Because q=0 of the kPCA latent maps to the data mean (NOT rest) for
*   one-sided motions, the reduced Vec1d state must be initialised to
*   q_rest = encode(0); the mapping itself uses the latent q as supplied.
*   See docs/methods/preimage_rom.md and component/kernel/PreImageProjector.h.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/mapping/ModelOrderReductionMapping.h>
#include <ModelOrderReduction/component/kernel/PreImageProjector.h>

#include <Eigen/Core>
#include <memory>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
class KernelPreImageMapping : public ModelOrderReductionMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(KernelPreImageMapping, TIn, TOut),
               SOFA_TEMPLATE2(ModelOrderReductionMapping, TIn, TOut));

    using Parent = ModelOrderReductionMapping<TIn, TOut>;
    using typename Parent::VecCoord;
    using typename Parent::VecDeriv;
    using typename Parent::InVecCoord;
    using typename Parent::InVecDeriv;
    using typename Parent::Coord;
    using typename Parent::Deriv;
    using typename Parent::InCoord;
    using typename Parent::InDeriv;
    using typename Parent::MatrixDeriv;
    using typename Parent::InMatrixDeriv;

    sofa::core::objectmodel::DataFileName d_kernelBundle;
    Data<double> d_eta;        ///< override bundle η  (<0 ⇒ keep bundle value)
    Data<double> d_etaT;       ///< override bundle η_t (<0 ⇒ keep bundle value)
    Data<int>    d_localRank;  ///< override bundle local-POD r (<0 ⇒ keep)
    Data<double> d_geomEps;    ///< FD step for applyDJT geometric stiffness

protected:
    KernelPreImageMapping();
    ~KernelPreImageMapping() override = default;

    std::unique_ptr<sofa::component::kernel::PreImageProjector> m_proj;

    // Previous step's converged pre-image displacement (η_t coherence anchor
    // and fixed-point warm reference). Displacement frame (u = X − X0).
    Eigen::VectorXd m_u_prev;

    // Per-step J(q) cache (3N × m). Built lazily by ensureJ from the current
    // parent latent q, reused by every applyJ / applyJT / applyJT(constraint)
    // in the step, invalidated at the end of apply (q changed).
    Eigen::MatrixXd m_J_cached;
    bool m_J_dirty = true;
    Eigen::VectorXd m_q_cached;   ///< latent at which m_J_cached was built (applyDJT FD)

    // apply() decodes u = Ψ(q) via the full fixed-point solve; cache it so
    // ensureJ builds J at that exact pre-image (jacobianAt) instead of
    // re-solving via the slower reduced Newton — the per-step cost driver.
    // m_have_decoded guards the first ensureJ that may precede any apply.
    Eigen::VectorXd m_u_decoded;  ///< Ψ(q) from the last apply()
    Eigen::VectorXd m_q_decoded;  ///< latent q that m_u_decoded was decoded from
    bool m_have_decoded = false;

    /// Rebuild m_J_cached = J(q) from the current fromModel latent if dirty.
    void ensureJ();
    /// Read the parent (reduced) latent q from fromModel.
    Eigen::VectorXd readLatent() const;

public:
    void init() override;
    void reset() override;

    void apply(const core::MechanicalParams* mparams,
               Data<VecCoord>& out, const Data<InVecCoord>& in) override;
    void applyJ(const core::MechanicalParams* mparams,
                Data<VecDeriv>& out, const Data<InVecDeriv>& in) override;
    void applyJT(const core::MechanicalParams* mparams,
                 Data<InVecDeriv>& out, const Data<VecDeriv>& in) override;
    void applyDJT(const core::MechanicalParams* mparams,
                  core::MultiVecDerivId parentForce,
                  core::ConstMultiVecDerivId childForce) override;
    void applyJT(const core::ConstraintParams* cparams,
                 Data<InMatrixDeriv>& out, const Data<MatrixDeriv>& in) override;
};

#if !defined(SOFA_COMPONENT_MAPPING_KERNELPREIMAGEMAPPING_CPP)
extern template class SOFA_MODELORDERREDUCTION_API
    KernelPreImageMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::mapping

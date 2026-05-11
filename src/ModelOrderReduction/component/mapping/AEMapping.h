/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   Autoencoder counterpart of ModelOrderReductionMapping.
*
*   Loads an AE bundle (X0, col_std, decoder.ts.pt, optional encoder.ts.pt
*   and rigid_modes.txt) and uses the latent-space decoder Jacobian J(q)
*   = ∂Ψ_θ/∂q for apply / applyJ / applyJT.
*
*   apply: incremental decode  u_new = u_old + J(q_old) · Δq.
*   applyJ:    du = J(q) · dq.
*   applyJT:   dq = J(q)ᵀ · du.
*
*   Parallel sibling to KernelPCAMapping; no edits to the kPCA path.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/mapping/ModelOrderReductionMapping.h>
#include <ModelOrderReduction/component/kernel/AEProjector.h>

#include <Eigen/Core>
#include <memory>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
class AEMapping : public ModelOrderReductionMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(AEMapping, TIn, TOut),
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

    sofa::core::objectmodel::DataFileName d_aeBundle;

protected:
    AEMapping();
    ~AEMapping() override = default;

    std::unique_ptr<sofa::component::kernel::AEProjector> m_projector;

    // Cached previous reduced coordinates for the incremental apply.
    // Reset to zero in init(); read/written in apply(). When the bundle has
    // rigid modes (nbRigid > 0), q is partitioned as [q_t (nbRigid); q_def (nbDef)].
    Eigen::VectorXd m_q_prev;

    // Cached translation columns from the bundle (3N × nbRigid). Empty when
    // nbRigid == 0; otherwise constant for the lifetime of the mapping.
    Eigen::MatrixXd m_PhiT;

    // Per-step dense J(q) cache. SOFA fires apply/applyJ/applyJT many times
    // per step (CG iterations, constraint solves). Building J once and doing
    // dense matmul / sparse indexing per call dominates direct JVP/VJP from
    // the projector — see project_ae_stage_f_revert: at m=78 trunk, direct
    // path was 70 ms/step vs cached at 40 ms/step. AEProjector still exposes
    // direct primitives for callers with low call counts.
    Eigen::MatrixXd m_J_cached;
    bool m_J_dirty = true;
    void ensureJ();

public:
    void init() override;
    void reset() override;

    void apply(const core::MechanicalParams* mparams,
               Data<VecCoord>& out, const Data<InVecCoord>& in) override;
    void applyJ(const core::MechanicalParams* mparams,
                Data<VecDeriv>& out, const Data<InVecDeriv>& in) override;
    void applyJT(const core::MechanicalParams* mparams,
                 Data<InVecDeriv>& out, const Data<VecDeriv>& in) override;
    void applyJT(const core::ConstraintParams* cparams,
                 Data<InMatrixDeriv>& out, const Data<MatrixDeriv>& in) override;
};

#if !defined(SOFA_COMPONENT_MAPPING_AEMAPPING_CPP)
extern template class SOFA_MODELORDERREDUCTION_API
    AEMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::mapping

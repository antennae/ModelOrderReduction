/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   kPCA counterpart of ModelOrderReductionMapping.
*
*   Loads a kPCA bundle (X0, snapshots D, dual coefficients α, kernel) and
*   uses the state-dependent decoder Jacobian J(u) = G(u)^{-1} J̃(u) for
*   apply / applyJ / applyJT, per `mor_kpca.md` §1–2.
*
*   apply: incremental decode  u_new = u_old + J(u_old) · Δq, with Δq taken
*          against a cached previous q (mor_kpca.md §4 option 1). For the
*          linear kernel J is constant and this collapses to the linear-ROM
*          formula u = X0 + Φ_eff · q.
*   applyJ:    du = J(u) · dq             — u read fresh from toModel.
*   applyJT:   dq = J(u)ᵀ · du = project_force(u, du)
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/mapping/ModelOrderReductionMapping.h>
#include <ModelOrderReduction/component/kernel/KernelProjector.h>

#include <Eigen/Core>
#include <memory>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
class KernelPCAMapping : public ModelOrderReductionMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(KernelPCAMapping, TIn, TOut),
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

protected:
    KernelPCAMapping();
    ~KernelPCAMapping() override = default;

    std::unique_ptr<sofa::component::kernel::KernelProjector> m_projector;

    // Cached previous reduced coordinates for the incremental apply.
    // Reset to zero in init() and reset(); read/written in apply().
    // When the bundle has rigid modes (nbRigid > 0), q is partitioned as
    // [q_t (nbRigid); q_def (nbDef)] and m_q_prev has size nbRigid+nbDef.
    Eigen::VectorXd m_q_prev;

    // Cached translation columns from the bundle (3N × nbRigid). Empty
    // when nbRigid == 0; otherwise constant for the lifetime of the mapping.
    Eigen::MatrixXd m_PhiT;

    // J(u) cache. Within a single animation step, u is constant from
    // MechanicalVInitVisitor through ConstraintSolver, so the same J(u) is
    // valid for every applyJT/applyJ/getJ call. SOFA's matrix-projection
    // path calls applyJT(constraint) repeatedly to build the projected
    // mass and stiffness; without this cache each call rebuilds J from
    // scratch (3N·T kernel evaluations for RBF). For the linear kernel J
    // is state-independent and cached once in init().
    Eigen::MatrixXd m_J_cached;
    bool m_J_dirty = true;
    bool m_J_constant = false;

    /// Rebuild m_J_cached from the current toModel position if m_J_dirty.
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

#if !defined(SOFA_COMPONENT_MAPPING_KERNELPCAMAPPING_CPP)
extern template class SOFA_MODELORDERREDUCTION_API
    KernelPCAMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::mapping

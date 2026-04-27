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
    Eigen::VectorXd m_q_prev;

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

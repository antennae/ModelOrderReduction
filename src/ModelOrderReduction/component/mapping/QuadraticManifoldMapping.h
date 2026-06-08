/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   Explicit quadratic-manifold decoder mapping.                                 *
******************************************************************************/
#pragma once

#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/QuadraticManifoldProjector.h>
#include <ModelOrderReduction/component/mapping/ModelOrderReductionMapping.h>

#include <Eigen/Core>
#include <memory>

namespace sofa::component::mapping
{

template <class TIn, class TOut>
class QuadraticManifoldMapping : public ModelOrderReductionMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(QuadraticManifoldMapping, TIn, TOut),
               SOFA_TEMPLATE2(ModelOrderReductionMapping, TIn, TOut));

    using Parent = ModelOrderReductionMapping<TIn, TOut>;
    using typename Parent::VecCoord;
    using typename Parent::VecDeriv;
    using typename Parent::InVecCoord;
    using typename Parent::InVecDeriv;
    using typename Parent::Coord;
    using typename Parent::Deriv;
    using typename Parent::InDeriv;
    using typename Parent::MatrixDeriv;
    using typename Parent::InMatrixDeriv;

    sofa::core::objectmodel::DataFileName d_quadraticManifoldBundle;

protected:
    QuadraticManifoldMapping();
    ~QuadraticManifoldMapping() override = default;

    std::unique_ptr<sofa::component::kernel::QuadraticManifoldProjector> m_projector;
    Eigen::VectorXd m_q_prev;
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

#if !defined(SOFA_COMPONENT_MAPPING_RESIDUALKERNELMAPPING_CPP)
extern template class SOFA_MODELORDERREDUCTION_API
    QuadraticManifoldMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::mapping

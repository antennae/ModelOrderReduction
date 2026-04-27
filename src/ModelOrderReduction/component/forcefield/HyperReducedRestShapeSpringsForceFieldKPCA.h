/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   kPCA counterpart of HyperReducedRestShapeSpringsForceField.
*   Mirrors the linear sibling but uses HyperReducedHelperKPCA so Gie rows
*   are computed via the state-dependent kPCA projection.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperKPCA.h>
#include <sofa/component/solidmechanics/spring/RestShapeSpringsForceField.h>

namespace sofa::core::behavior
{
template< class T > class MechanicalState;
} // namespace sofa::core::behavior

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
class HyperReducedRestShapeSpringsForceFieldKPCA
    : public virtual RestShapeSpringsForceField<DataTypes>
    , public modelorderreduction::HyperReducedHelperKPCA
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HyperReducedRestShapeSpringsForceFieldKPCA, DataTypes),
                SOFA_TEMPLATE(RestShapeSpringsForceField, DataTypes),
                HyperReducedHelperKPCA);

    typedef HyperReducedHelperKPCA Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector< sofa::Index > VecIndex;
    typedef type::vector< Real > VecReal;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    ////////////////////////// Inherited attributes ////////////////////////////

    using HyperReducedHelperKPCA::d_prepareECSW;
    using HyperReducedHelperKPCA::d_kernelBundle;
    using HyperReducedHelperKPCA::d_nbTrainingSet;
    using HyperReducedHelperKPCA::d_periodSaveGIE;

    using HyperReducedHelperKPCA::d_performECSW;
    using HyperReducedHelperKPCA::d_RIDPath;
    using HyperReducedHelperKPCA::d_weightsPath;

    using HyperReducedHelperKPCA::Gie;
    using HyperReducedHelperKPCA::weights;
    using HyperReducedHelperKPCA::reducedIntegrationDomain;

    using HyperReducedHelperKPCA::m_projector;
    using HyperReducedHelperKPCA::m_RIDsize;


    using RestShapeSpringsForceField<DataTypes>::d_points;
    using RestShapeSpringsForceField<DataTypes>::d_stiffness;
    using RestShapeSpringsForceField<DataTypes>::d_angularStiffness;
    using RestShapeSpringsForceField<DataTypes>::d_pivotPoints;
    using RestShapeSpringsForceField<DataTypes>::d_external_points;
    using RestShapeSpringsForceField<DataTypes>::d_recompute_indices;
    using RestShapeSpringsForceField<DataTypes>::d_drawSpring;
    using RestShapeSpringsForceField<DataTypes>::d_springColor;
    using RestShapeSpringsForceField<DataTypes>::l_restMState;
    using RestShapeSpringsForceField<DataTypes>::d_activeDirections;
    using RestShapeSpringsForceField<DataTypes>::matS;

protected:
    HyperReducedRestShapeSpringsForceFieldKPCA();

public:
    void bwdInit() override;
    virtual void parse(core::objectmodel::BaseObjectDescription *arg) override;
    virtual void reinit() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* /* matrix */) override;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    const DataVecCoord* getExtPosition() const;
    const VecIndex& getExtIndices() const { return (useRestMState ? m_ext_indices : m_indices); }

protected:
    void recomputeIndices();
    bool checkOutOfBoundsIndices();

    using RestShapeSpringsForceField<DataTypes>::m_indices;
    using RestShapeSpringsForceField<DataTypes>::m_ext_indices;
    using RestShapeSpringsForceField<DataTypes>::m_pivots;

    using RestShapeSpringsForceField<DataTypes>::lastUpdatedStep;

private:
    bool useRestMState;
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDRESTSHAPESPRINGSFORCEFIELDKPCA_CPP)
extern template class SOFA_MODELORDERREDUCTION_API HyperReducedRestShapeSpringsForceFieldKPCA<sofa::defaulttype::Vec3Types>;
#endif
} // namespace sofa::component::solidmechanics::spring

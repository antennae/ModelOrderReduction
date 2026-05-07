/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   AE counterpart of HyperReducedRestShapeSpringsForceFieldKPCA.
*   Mirrors the kPCA sibling but uses HyperReducedHelperAE so Gie rows
*   are computed via the AE decoder Jacobian J(q).
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperAE.h>
#include <sofa/component/solidmechanics/spring/RestShapeSpringsForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::core::behavior
{
template< class T > class MechanicalState;
} // namespace sofa::core::behavior

namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
class HyperReducedRestShapeSpringsForceFieldAE
    : public virtual RestShapeSpringsForceField<DataTypes>
    , public modelorderreduction::HyperReducedHelperAE
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HyperReducedRestShapeSpringsForceFieldAE, DataTypes),
                SOFA_TEMPLATE(RestShapeSpringsForceField, DataTypes),
                HyperReducedHelperAE);

    typedef HyperReducedHelperAE Inherit;
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

    using HyperReducedHelperAE::d_prepareECSW;
    using HyperReducedHelperAE::d_aeBundle;
    using HyperReducedHelperAE::d_nbTrainingSet;
    using HyperReducedHelperAE::d_periodSaveGIE;

    using HyperReducedHelperAE::d_performECSW;
    using HyperReducedHelperAE::d_RIDPath;
    using HyperReducedHelperAE::d_weightsPath;

    using HyperReducedHelperAE::Gie;
    using HyperReducedHelperAE::weights;
    using HyperReducedHelperAE::reducedIntegrationDomain;

    using HyperReducedHelperAE::m_projector;
    using HyperReducedHelperAE::m_RIDsize;


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

    // Cached pointer to the Vec1d MechanicalObject holding q. Resolved at
    // bwdInit() by walking up the scene graph.
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* m_qState = nullptr;

protected:
    HyperReducedRestShapeSpringsForceFieldAE();

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

#if !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDRESTSHAPESPRINGSFORCEFIELDAE_CPP)
extern template class SOFA_MODELORDERREDUCTION_API HyperReducedRestShapeSpringsForceFieldAE<sofa::defaulttype::Vec3Types>;
#endif
} // namespace sofa::component::solidmechanics::spring

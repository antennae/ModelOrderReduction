/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   AE counterpart of HyperReducedTriangleFEMForceFieldKPCA — triangle
*   sibling of HyperReducedTetrahedronFEMForceFieldAE. Used for the
*   PDMS membrane sub-topology in the multiGait quadruped.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperAE.h>
#include <sofa/component/solidmechanics/fem/elastic/TriangleFEMForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::forcefield
{
namespace {
using sofa::component::solidmechanics::fem::elastic::TriangleFEMForceField;
}


template<class DataTypes>
class HyperReducedTriangleFEMForceFieldAE
    : public virtual TriangleFEMForceField<DataTypes>
    , public modelorderreduction::HyperReducedHelperAE
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HyperReducedTriangleFEMForceFieldAE, DataTypes),
                SOFA_TEMPLATE(TriangleFEMForceField, DataTypes),
                HyperReducedHelperAE);

    typedef core::behavior::ForceField<DataTypes> Inherited;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord    Coord;
    typedef typename DataTypes::Deriv    Deriv;
    typedef typename Coord::value_type   Real;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef sofa::Index Index;
    typedef sofa::core::topology::BaseMeshTopology::Triangle Element;
    typedef sofa::core::topology::BaseMeshTopology::SeqTriangles VecElement;

    static const int SMALL = 1;
    static const int LARGE = 0;

    // Cached pointer to the Vec1d MechanicalObject holding q. Resolved at
    // init() time by walking up the scene graph.
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* m_qState = nullptr;

protected:
    typedef type::Vec<6, Real> Displacement;
    typedef type::Mat<3, 3, Real> MaterialStiffness;
    typedef sofa::type::vector<MaterialStiffness> VecMaterialStiffness;
    typedef type::Mat<6, 3, Real> StrainDisplacement;
    typedef sofa::type::vector<StrainDisplacement> VecStrainDisplacement;
    typedef type::Mat<3, 3, Real> Transformation;
    typedef type::Mat<9, 9, Real> StiffnessMatrix;

    using TriangleFEMForceField<DataTypes>::_materialsStiffnesses;
    using TriangleFEMForceField<DataTypes>::_strainDisplacements;
    using TriangleFEMForceField<DataTypes>::_indexedElements;
    using TriangleFEMForceField<DataTypes>::d_initialPoints;
    using TriangleFEMForceField<DataTypes>::_rotatedInitialElements;
    using TriangleFEMForceField<DataTypes>::_rotations;

public:
    void init() override;
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    SReal getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const override
    {
        msg_error() << "Get potentialEnergy not implemented";
        return 0.0;
    }

    void draw(const core::visual::VisualParams* vparams) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    using TriangleFEMForceField<DataTypes>::method;
    using TriangleFEMForceField<DataTypes>::d_method;
    using TriangleFEMForceField<DataTypes>::d_poissonRatio;
    using TriangleFEMForceField<DataTypes>::d_youngModulus;
    using TriangleFEMForceField<DataTypes>::d_thickness;
    using TriangleFEMForceField<DataTypes>::d_planeStrain;

protected:
    void hyperReducedAccumulateForceLarge(VecCoord& f, const VecCoord& p, bool implicit);
    void hyperReducedApplyStiffnessLarge(VecCoord& f, Real h, const VecCoord& x, const SReal& kFactor);
};


#if !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTRIANGLEFEMFORCEFIELDAE_CPP)
extern template class SOFA_MODELORDERREDUCTION_API HyperReducedTriangleFEMForceFieldAE<sofa::defaulttype::Vec3Types>;
#endif

} // namespace sofa::component::forcefield

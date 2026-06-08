/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   quadratic-manifold counterpart of HyperReducedTetrahedronFEMForceFieldKPCA.
*   Same physics + ECSW handling; the only behavioral differences are
*   (a) inheritance from HyperReducedHelperQuadraticManifold, and (b) addForce calls
*   prepareFrame(q_def) — reading the latent state from the parent Vec1d
*   MechanicalObject — once before the element loop. Gie collection uses
*   the quadratic-manifold helper's updateGie.
*
*   Runtime: option B (mapping-projects). The element loop writes
*   ξ_e · f_e to the Vec3d output and QuadraticManifoldMapping::applyJT contracts to q
*   in one VJP. Algebraically equivalent to per-element J^T projection
*   but cheaper because we don't rebuild J(q) per element.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/forcefield/HyperReducedHelperQuadraticManifold.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>

#define SIMPLEFEM_COLORMAP
#define SOFAHYPERREDUCEDTETRAHEDRONFEMFORCEFIELDRESIDUALKERNEL_COLORMAP

#include <sofa/helper/ColorMap.h>


namespace sofa::component::forcefield
{
namespace { using sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField; }


template<class DataTypes>
class HyperReducedTetrahedronFEMForceFieldQuadraticManifold
    : public virtual TetrahedronFEMForceField<DataTypes>
    , public modelorderreduction::HyperReducedHelperQuadraticManifold
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(HyperReducedTetrahedronFEMForceFieldQuadraticManifold, DataTypes),
                SOFA_TEMPLATE(TetrahedronFEMForceField, DataTypes),
                HyperReducedHelperQuadraticManifold);

    typedef typename core::behavior::ForceField<DataTypes> InheritForceField;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef sofa::Index Index;
    typedef core::topology::BaseMeshTopology::Tetra Element;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra VecElement;
    typedef core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
    using index_type = sofa::Index;

    enum { SMALL = 0, LARGE = 1, POLAR = 2, SVD = 3 };

    // Cached pointer to the Vec1d MechanicalObject holding q. Only needed
    // during Gie collection (prepareECSW); null at runtime. Resolved at
    // init() by walking up the scene graph for the Vec1d state whose size
    // matches nbRigid+nbDef — robust to scene-graph rewiring done by
    // `modifyGraphSceneQuadraticManifold`, and to cross-parented Rigidify/articulated
    // topologies where smaller Vec1d (servo angle) states also exist.
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* m_qState = nullptr;

protected:

    typedef type::VecNoInit<12, Real> Displacement;
    typedef type::Mat<6, 6, Real> MaterialStiffness;
    typedef type::Mat<12, 6, Real> StrainDisplacement;

    type::MatNoInit<3, 3, Real> R0;

    typedef type::MatNoInit<3, 3, Real> Transformation;
    typedef type::Mat<12, 12, Real> StiffnessMatrix;
    typedef type::VecNoInit<6,Real> VoigtTensor;

    typedef type::vector<MaterialStiffness> VecMaterialStiffness;
    typedef type::vector<StrainDisplacement> VecStrainDisplacement;

    typedef type::Mat<4, 4, Real> Mat44;
    typedef type::Mat<3, 3, Real> Mat33;
    typedef type::Mat<4, 3, Real> Mat43;

    using TetrahedronFEMForceField<DataTypes>::materialsStiffnesses;
    using TetrahedronFEMForceField<DataTypes>::strainDisplacements;
    using TetrahedronFEMForceField<DataTypes>::rotations;
    using TetrahedronFEMForceField<DataTypes>::_plasticStrains;
    using TetrahedronFEMForceField<DataTypes>::d_showVonMisesStressPerElement;

    typedef std::pair<index_type,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;

public:

    HyperReducedTetrahedronFEMForceFieldQuadraticManifold();

    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;

    using TetrahedronFEMForceField<DataTypes>::getPotentialEnergy;

    virtual void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void draw(const core::visual::VisualParams* vparams) override;

protected:

    void accumulateForceSmall( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );
    void applyStiffnessSmall( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0  );

    using TetrahedronFEMForceField<DataTypes>::_rotatedInitialElements;
    using TetrahedronFEMForceField<DataTypes>::_initialRotations;
    void accumulateForceLarge( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    using TetrahedronFEMForceField<DataTypes>::_rotationIdx;
    void accumulateForcePolar( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    using TetrahedronFEMForceField<DataTypes>::_initialTransformation;
    void accumulateForceSVD( Vector& f, const Vector & p, typename VecElement::const_iterator elementIt, Index elementIndex );

    void applyStiffnessCorotational( Vector& f, const Vector& x, int i=0, Index a=0,Index b=1,Index c=2,Index d=3, SReal fact=1.0  );
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONFEMFORCEFIELDRESIDUALKERNEL_CPP)
extern template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronFEMForceFieldQuadraticManifold<defaulttype::Vec3Types>;
#endif
} // namespace sofa::component::forcefield

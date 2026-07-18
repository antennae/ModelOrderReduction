/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   quadratic-manifold counterpart of HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA.
*   Same physics + edge-based tangent + ECSW handling; the only behavioral
*   differences are (a) inheritance from HyperReducedHelperQuadraticManifold, and (b)
*   addForce calls prepareFrame(q_def) — reading the latent state from the
*   parent Vec1d MechanicalObject — once before the element loop. Gie
*   collection uses the quadratic-manifold helper's updateGie.
*
*   Purpose: test whether a hyperelastic material law needs the quadratic-manifold
*   (decoder-Jacobian) ROM, alongside the kPCA sibling.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperQuadraticManifold.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/TetrahedronHyperelasticityFEMForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::forcefield
{
using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using namespace solidmechanics::fem::hyperelastic;

template<class DataTypes>
class HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold
    : public virtual solidmechanics::fem::hyperelastic::TetrahedronHyperelasticityFEMForceField<DataTypes>
    , public modelorderreduction::HyperReducedHelperQuadraticManifold
{
  public:
    SOFA_CLASS2(SOFA_TEMPLATE(HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold, DataTypes),
        SOFA_TEMPLATE(solidmechanics::fem::hyperelastic::TetrahedronHyperelasticityFEMForceField, DataTypes),
        modelorderreduction::HyperReducedHelperQuadraticManifold);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename TetrahedronHyperelasticityFEMForceField<DataTypes>::EdgeInformation EdgeInformation;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;

    typedef type::Mat<3,3,Real> Matrix3;
    typedef type::MatSym<3,Real> MatrixSym;
    typedef std::pair<MatrixSym,MatrixSym> MatrixPair;
    typedef std::pair<Real,MatrixSym> MatrixCoeffPair;


    typedef type::vector<Real> SetParameterArray;
    typedef type::vector<Coord> SetAnisotropyDirectionArray;


    typedef sofa::Index Index;
    typedef core::topology::BaseMeshTopology::Tetra Element;
    typedef core::topology::BaseMeshTopology::SeqTetrahedra VecElement;
    typedef sofa::core::topology::Topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::Topology::TetraID TetraID;
    typedef sofa::core::topology::Topology::Tetra Tetra;
    typedef sofa::core::topology::Topology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::EdgesInTriangle EdgesInTriangle;
    typedef sofa::core::topology::BaseMeshTopology::EdgesInTetrahedron EdgesInTetrahedron;
    typedef sofa::core::topology::BaseMeshTopology::TrianglesInTetrahedron TrianglesInTetrahedron;


public :

    using TetrahedronHyperelasticityFEMForceField<DataTypes>::globalParameters;

    // Cached pointer to the Vec1d MechanicalObject holding q. Only needed
    // during Gie collection (prepareECSW); null at runtime. Resolved at
    // init() by walking up the scene graph for the Vec1d state whose size
    // matches nbRigid+nbDef — see the quadratic-manifold linear-tetra sibling for the size
    // filter rationale.
    sofa::core::behavior::MechanicalState<sofa::defaulttype::Vec1Types>* m_qState = nullptr;


 protected :
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_topology;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_initialPoints;	/// the intial positions of the points
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_updateMatrix;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_stiffnessMatrixRegularizationWeight;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_materialName; /// the name of the material
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_parameterSet;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_anisotropySet;

    // Reduced Order Variables (quadratic-manifold helper — no d_nbModes/d_modesPath; the
    // quadratic-manifold bundle fully determines the mode count).

    using HyperReducedHelperQuadraticManifold::d_prepareECSW;
    using HyperReducedHelperQuadraticManifold::d_quadraticManifoldBundle;
    using HyperReducedHelperQuadraticManifold::d_nbTrainingSet;
    using HyperReducedHelperQuadraticManifold::d_periodSaveGIE;

    using HyperReducedHelperQuadraticManifold::d_performECSW;
    using HyperReducedHelperQuadraticManifold::d_RIDPath;
    using HyperReducedHelperQuadraticManifold::d_weightsPath;

    using HyperReducedHelperQuadraticManifold::Gie;
    using HyperReducedHelperQuadraticManifold::weights;
    using HyperReducedHelperQuadraticManifold::reducedIntegrationDomain;

    using HyperReducedHelperQuadraticManifold::m_nbModes;
    using HyperReducedHelperQuadraticManifold::m_RIDsize;
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> reducedIntegrationDomainWithEdges;

    unsigned int m_RIDedgeSize;

    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_tetrahedronInfo;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_edgeInfo;

protected:
   HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold();

   virtual   ~HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold();
public:

    virtual void init() override;

    virtual void addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    virtual void addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* /* matrix */) override;

    void draw(const core::visual::VisualParams* vparams) override;

  protected:

    /// the array that describes the complete material energy and its derivatives

    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_myMaterial;

    void updateTangentMatrix();
};

using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

#if !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDQUADRATICMANIFOLD_CPP)
extern template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronHyperelasticityFEMForceFieldQuadraticManifold<Vec3Types>;
#endif // !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDQUADRATICMANIFOLD_CPP)
} // namespace sofa::component::forcefield

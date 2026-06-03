/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   kPCA counterpart of HyperReducedTetrahedronHyperelasticityFEMForceField.
*   Mirrors that sibling exactly in physics (deformation gradient, SPK
*   tensor, edge-based tangent) and ECSW handling; the only behavioral
*   difference is the per-frame state-dependent projection — addForce calls
*   HyperReducedHelperKPCA::prepareFrame(u) once before the element loop,
*   and Gie collection uses the kPCA helper's updateGie.
*
*   Purpose: test whether a hyperelastic material law (Neo-Hookean,
*   Mooney-Rivlin, ...) needs the non-linear (kPCA/AE) ROM, or whether the
*   linear-POD reduction already resolves it — the same question we answered
*   for the corotational law via the linear-tetra siblings.
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperKPCA.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/TetrahedronHyperelasticityFEMForceField.h>

namespace sofa::component::forcefield
{
using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;
using namespace solidmechanics::fem::hyperelastic;

template<class DataTypes>
class HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA
    : public virtual solidmechanics::fem::hyperelastic::TetrahedronHyperelasticityFEMForceField<DataTypes>
    , public modelorderreduction::HyperReducedHelperKPCA
{
  public:
    SOFA_CLASS2(SOFA_TEMPLATE(HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA, DataTypes),
        SOFA_TEMPLATE(solidmechanics::fem::hyperelastic::TetrahedronHyperelasticityFEMForceField, DataTypes),
        modelorderreduction::HyperReducedHelperKPCA);

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


 protected :
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_topology;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_initialPoints;	/// the intial positions of the points
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_updateMatrix;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_stiffnessMatrixRegularizationWeight;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_materialName; /// the name of the material
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_parameterSet;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::d_anisotropySet;

    // Reduced Order Variables (kPCA helper — no d_nbModes/d_modesPath; the
    // kernel bundle fully determines the mode count).

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

    using HyperReducedHelperKPCA::m_nbModes;
    using HyperReducedHelperKPCA::m_RIDsize;
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> reducedIntegrationDomainWithEdges;

    unsigned int m_RIDedgeSize;

    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_tetrahedronInfo;
    using TetrahedronHyperelasticityFEMForceField<DataTypes>::m_edgeInfo;

protected:
   HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA();

   virtual   ~HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA();
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

#if !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDKPCA_CPP)
extern template class SOFA_MODELORDERREDUCTION_API HyperReducedTetrahedronHyperelasticityFEMForceFieldKPCA<Vec3Types>;
#endif // !defined(SOFA_COMPONENT_FORCEFIELD_HYPERREDUCEDTETRAHEDRONHYPERELASTICITYFEMFORCEFIELDKPCA_CPP)
} // namespace sofa::component::forcefield

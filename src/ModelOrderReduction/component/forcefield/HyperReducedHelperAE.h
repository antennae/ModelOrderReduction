/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   AE counterpart of HyperReducedHelperKPCA: collects Gie rows using a
*   state-dependent Galerkin projection J(q) = ∂Ψ_θ/∂q.
*   Loads an AE bundle via AEProjector; shares Gie / ECSW bookkeeping
*   shape with the kPCA helper (parallel sibling — no inheritance).
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/AEProjector.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>


namespace modelorderreduction
{

using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::Data;

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperAE
    : public virtual BaseObject
{
public:
    SOFA_CLASS(HyperReducedHelperAE, BaseObject);

    Data<bool>         d_prepareECSW;
    Data<std::string>  d_aeBundle;
    Data<unsigned int> d_nbTrainingSet;
    Data<unsigned int> d_periodSaveGIE;

    Data<bool>         d_performECSW;
    Data<std::string>  d_RIDPath;
    Data<std::string>  d_weightsPath;

    // Loaded AE bundle.
    std::unique_ptr<sofa::component::kernel::AEProjector> m_projector;
    // Augmented mode count: nbRigid + nbDef.
    unsigned int m_nbModes = 0;
    Eigen::MatrixXd m_PhiT;       // (3N, nbRigid); empty if 0
    unsigned int m_nbRigid = 0;
    unsigned int m_nbDef = 0;

    // Per-frame cache — refreshed by prepareFrame(q_def), consumed by updateGie.
    Eigen::MatrixXd m_J_def;      // (3N, m_def)  J(q_def) from AEProjector
    bool m_frameReady = false;

    // Gie bookkeeping (matches HyperReducedHelperKPCA so phase 4 reads it
    // back unchanged).
    std::vector<std::vector<double>> Gie;
    Eigen::VectorXd  weights;
    Eigen::VectorXi  reducedIntegrationDomain;
    unsigned int     m_RIDsize = 0;

    HyperReducedHelperAE();

    /// Load bundle, allocate Gie, load ECSW weights/RID if performECSW.
    void initMOR(unsigned int nbElements, bool printLog = false);

    /// Refresh the per-frame Jacobian cache. `q_def` is the deformation
    /// part of the latent state (size m_def). Call once per addForce
    /// before the element loop.
    void prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& q_def);

    /// Compute [J(q)^T f_e]_k for one element. Caller must have called
    /// prepareFrame(q_def) first.
    template <class DataTypes>
    Eigen::VectorXd projectOneElement(
        const std::vector<unsigned int>&              indexList,
        const std::vector<typename DataTypes::Deriv>& contrib) const;

    /// Accumulate one element's contribution into Gie.
    template <class DataTypes>
    void updateGie(const std::vector<unsigned int>&              indexList,
                   const std::vector<typename DataTypes::Deriv>& contrib,
                   unsigned int                                  numElem);

    void saveGieFile(unsigned int nbElements);
};


template <class DataTypes>
Eigen::VectorXd HyperReducedHelperAE::projectOneElement(
    const std::vector<unsigned int>&              indexList,
    const std::vector<typename DataTypes::Deriv>& contrib) const
{
    const unsigned V = static_cast<unsigned>(indexList.size());

    Eigen::VectorXd GieUnit(m_nbModes);

    // Rigid rows: Φ_t[elem_dofs, :]ᵀ · contrib (no J_def involvement).
    for (unsigned k = 0; k < m_nbRigid; ++k)
    {
        double s = 0.0;
        for (unsigned i = 0; i < V; ++i)
        {
            const unsigned dof0 = 3 * indexList[i];
            s += m_PhiT(dof0 + 0, k) * contrib[i][0]
               + m_PhiT(dof0 + 1, k) * contrib[i][1]
               + m_PhiT(dof0 + 2, k) * contrib[i][2];
        }
        GieUnit(k) = s;
    }

    // Deformation rows: J_def[elem_dofs, :]ᵀ · contrib.
    for (unsigned k = 0; k < m_nbDef; ++k)
    {
        double s = 0.0;
        for (unsigned i = 0; i < V; ++i)
        {
            const unsigned dof0 = 3 * indexList[i];
            s += m_J_def(dof0 + 0, k) * contrib[i][0]
               + m_J_def(dof0 + 1, k) * contrib[i][1]
               + m_J_def(dof0 + 2, k) * contrib[i][2];
        }
        GieUnit(m_nbRigid + k) = s;
    }
    return GieUnit;
}


template <class DataTypes>
void HyperReducedHelperAE::updateGie(
    const std::vector<unsigned int>&              indexList,
    const std::vector<typename DataTypes::Deriv>& contrib,
    unsigned int                                  numElem)
{
    if (!d_prepareECSW.getValue() || !m_frameReady)
        return;

    const int step = int(this->getContext()->getTime() / this->getContext()->getDt());
    if (step % d_periodSaveGIE.getValue() != 0)
        return;
    const unsigned numTest = step / d_periodSaveGIE.getValue();
    if (numTest >= d_nbTrainingSet.getValue())
        return;

    const Eigen::VectorXd GieUnit = projectOneElement<DataTypes>(indexList, contrib);
    for (unsigned k = 0; k < m_nbModes; ++k)
        Gie[m_nbModes * numTest + k][numElem] = GieUnit(k);
}

} // namespace modelorderreduction

/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   HyperReducedHelperDecoder — shared ECSW/Gie bookkeeping for the q-keyed   *
*   decoder helpers (AE / residual-kernel / quadratic-manifold). Holds a      *
*   DecoderProjector* and a per-frame dense J cache. Concrete shims supply    *
*   createProjector() + bundlePath(). The linear HyperReducedHelper and the   *
*   state-keyed HyperReducedHelperKPCA are deliberately NOT collapsed here.   *
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/DecoderProjector.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/type/vector.h>

#include <Eigen/Core>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace modelorderreduction
{

using sofa::core::objectmodel::BaseObject;
using sofa::core::objectmodel::Data;

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperDecoder
    : public virtual BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(HyperReducedHelperDecoder, BaseObject);

    // Shared training/reduction config (identical across all decoder methods;
    // the per-method bundle path lives in the concrete shim).
    Data<bool>         d_prepareECSW;
    Data<unsigned int> d_nbTrainingSet;
    Data<unsigned int> d_periodSaveGIE;
    Data<bool>         d_performECSW;
    Data<std::string>  d_RIDPath;
    Data<std::string>  d_weightsPath;
    Data<sofa::type::vector<int>> d_indexMap;

    // Loaded decoder + augmented mode counts (nbRigid + nbDef).
    std::unique_ptr<sofa::component::kernel::DecoderProjector> m_projector;
    unsigned int m_nbModes = 0;
    unsigned int m_nbDef   = 0;
    unsigned int m_nbRigid = 0;
    Eigen::MatrixXd m_PhiT;        // (3N, nbRigid); empty when none

    // Per-frame dense Jacobian cache (def block), refreshed by prepareFrame.
    Eigen::MatrixXd m_J;          // (3N, nbDef)
    bool m_frameReady = false;

    Eigen::VectorXi m_indexMap;   // resolved from d_indexMap at init

    // Gie / ECSW bookkeeping (same layout as the linear HyperReducedHelper).
    std::vector<std::vector<double>> Gie;
    Eigen::VectorXd  weights;
    Eigen::VectorXi  reducedIntegrationDomain;
    unsigned int     m_RIDsize = 0;

    /// Load bundle, allocate Gie, load ECSW weights/RID if performECSW.
    void initMOR(unsigned int nbElements, bool printLog = false);

    /// Refresh the per-frame dense Jacobian cache. Call once per addForce
    /// before the element loop. `q` is the latent state (size m_def).
    void prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& q);

    /// Compute [J(q)^T f_e]_k for one element. Caller must have called
    /// prepareFrame(q) first.
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

protected:
    HyperReducedHelperDecoder();

    /// Construct the concrete decoder projector (shim supplies the type).
    virtual std::unique_ptr<sofa::component::kernel::DecoderProjector>
    createProjector() const = 0;
    /// Bundle directory for loadFromBundle (shim returns its d_*Bundle value).
    virtual std::string bundlePath() const = 0;
};


template <class DataTypes>
Eigen::VectorXd HyperReducedHelperDecoder::projectOneElement(
    const std::vector<unsigned int>&              indexList,
    const std::vector<typename DataTypes::Deriv>& contrib) const
{
    const unsigned V = static_cast<unsigned>(indexList.size());

    // Resolve each element vertex into the bundle's deformable dof list.
    // -1 marks a rigid vertex (Rigidify) whose motion flows through a
    // RigidMapping, not the decoder.
    std::vector<long> defIdx(V);
    for (unsigned i = 0; i < V; ++i)
    {
        if (static_cast<long>(indexList[i]) >= m_indexMap.size())
            throw std::runtime_error(
                "HyperReducedHelperDecoder: vertex index " +
                std::to_string(indexList[i]) +
                " exceeds indexMap size " +
                std::to_string(m_indexMap.size()) + ".");
        defIdx[i] = m_indexMap(indexList[i]);
    }

    // Fail loud if any resolved bundle slot would index past the decoder's
    // full (3N) state — catches a corrupt indexMap/bundle here at the shared
    // convergence point rather than as silent release UB in the dot products.
    const unsigned nbStateDofs = m_projector->nbDofs();
    for (unsigned i = 0; i < V; ++i)
    {
        if (defIdx[i] < 0) continue;
        if (3u * static_cast<unsigned>(defIdx[i]) + 2u >= nbStateDofs)
            throw std::runtime_error(
                "HyperReducedHelperDecoder: indexMap slot " +
                std::to_string(defIdx[i]) + " exceeds decoder state size " +
                std::to_string(nbStateDofs) + ".");
    }

    Eigen::VectorXd GieUnit(m_nbModes);

    // Rigid rows: Phi_t[def_dofs, :]^T . contrib (constant decoder block).
    for (unsigned k = 0; k < m_nbRigid; ++k)
    {
        double s = 0.0;
        for (unsigned i = 0; i < V; ++i)
        {
            if (defIdx[i] < 0) continue;
            const unsigned dof0 = 3 * static_cast<unsigned>(defIdx[i]);
            s += m_PhiT(dof0 + 0, k) * contrib[i][0]
               + m_PhiT(dof0 + 1, k) * contrib[i][1]
               + m_PhiT(dof0 + 2, k) * contrib[i][2];
        }
        GieUnit(k) = s;
    }

    // Deformation rows: J[def_dofs, :]^T . contrib.
    for (unsigned k = 0; k < m_nbDef; ++k)
    {
        double s = 0.0;
        for (unsigned i = 0; i < V; ++i)
        {
            if (defIdx[i] < 0) continue;
            const unsigned dof0 = 3 * static_cast<unsigned>(defIdx[i]);
            s += m_J(dof0 + 0, k) * contrib[i][0]
               + m_J(dof0 + 1, k) * contrib[i][1]
               + m_J(dof0 + 2, k) * contrib[i][2];
        }
        GieUnit(m_nbRigid + k) = s;
    }
    return GieUnit;
}


template <class DataTypes>
void HyperReducedHelperDecoder::updateGie(
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

/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   kPCA counterpart of HyperReducedHelper: collects Gie rows using a
*   state-dependent Galerkin projection J(q) = G(u)^{-1} J̃(u).
*   Loads a kPCA bundle via KernelProjector; shares Gie / ECSW bookkeeping
*   shape with the linear-ROM HyperReducedHelper (deliberately independent
*   from it — no inheritance — to keep the linear path untouched).
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/KernelProjector.h>

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

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperKPCA
    : public virtual BaseObject
{
public:
    SOFA_CLASS(HyperReducedHelperKPCA, BaseObject);

    // Training & reduction configuration (same shape as HyperReducedHelper,
    // minus d_modesPath / d_nbModes — the bundle fully determines these).
    Data<bool>         d_prepareECSW;
    Data<std::string>  d_kernelBundle;
    Data<unsigned int> d_nbTrainingSet;
    Data<unsigned int> d_periodSaveGIE;

    Data<bool>         d_performECSW;
    Data<std::string>  d_RIDPath;
    Data<std::string>  d_weightsPath;

    // FF-mstate -> bundle-deformable index map. One entry per FF mstate
    // vertex; value is the slot in the bundle's deformable dofs, or -1 for
    // vertices that don't participate in the kPCA basis (e.g. Rigidify rigid-
    // group particles whose displacement flows through a RigidMapping). When
    // empty, the FF assumes the bundle covers every mstate vertex 1:1 and
    // uses indexList[i] directly — that's the legacy path; if it would OOB
    // m_grad_all / m_PhiT, projectOneElement throws.
    Data<sofa::type::vector<int>> d_indexMap;

    // Loaded kPCA bundle.
    std::unique_ptr<sofa::component::kernel::KernelProjector> m_projector;
    // Augmented mode count: nbRigid + nbDef. Sized so Gie has one row per
    // augmented mode at every training capture; phase 4 NNLS reads
    // m_nbModes rows per snapshot.
    unsigned int m_nbModes = 0;
    // Rigid columns from the bundle (3N × nbRigid). Empty when the bundle
    // has no rigid modes; otherwise the leading rows of every Gie capture
    // are Φ_t[elem_dofs, :]ᵀ · contrib (no G^{-1}, since Φ_t is constant).
    Eigen::MatrixXd m_PhiT;
    unsigned int m_nbRigid = 0;

    // Per-frame cache — refreshed by prepareFrame(u), consumed by updateGie.
    Eigen::MatrixXd m_grad_all;   // (3N, T)  grad_u(u, snapshots)
    double m_sigma2 = 1.0;        // cached G^{-1} scalar for RBF / linear
    bool m_frameReady = false;

    // Resolved from d_indexMap at init. Empty -> legacy direct-index path.
    Eigen::VectorXi m_indexMap;

    // Gie bookkeeping (matches HyperReducedHelper’s storage layout so phase 4
    // can consume the output unchanged).
    std::vector<std::vector<double>> Gie;
    Eigen::VectorXd  weights;
    Eigen::VectorXi  reducedIntegrationDomain;
    unsigned int     m_RIDsize = 0;

    HyperReducedHelperKPCA();

    /// Load bundle, allocate Gie, load ECSW weights/RID if performECSW.
    void initMOR(unsigned int nbElements, bool printLog = false);

    /// Refresh the per-frame grad cache. Call once per addForce before the
    /// element loop. `u` is the current state vector of shape (3N,).
    void prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& u);

    /// Compute [J(q)^T f_e]_k for one element — testable in isolation.
    /// Caller must have called prepareFrame(u) first.
    template <class DataTypes>
    Eigen::VectorXd projectOneElement(
        const std::vector<unsigned int>&              indexList,
        const std::vector<typename DataTypes::Deriv>& contrib) const;

    /// Accumulate one element's contribution into Gie.
    template <class DataTypes>
    void updateGie(const std::vector<unsigned int>&                 indexList,
                   const std::vector<typename DataTypes::Deriv>&    contrib,
                   unsigned int                                     numElem);

    void saveGieFile(unsigned int nbElements);
};


template <class DataTypes>
Eigen::VectorXd HyperReducedHelperKPCA::projectOneElement(
    const std::vector<unsigned int>&              indexList,
    const std::vector<typename DataTypes::Deriv>& contrib) const
{
    const unsigned V = static_cast<unsigned>(indexList.size());
    const unsigned T = m_projector->nbSnapshots();
    const unsigned mDef = m_projector->nbModes();

    // G^{-1}·contrib — element-local. G^{-1} = σ²·I for both linear (σ²=1) and RBF.
    Eigen::VectorXd Ginv_contrib(3 * V);
    for (unsigned i = 0; i < V; ++i)
        for (unsigned c = 0; c < 3; ++c)
            Ginv_contrib(3 * i + c) = m_sigma2 * contrib[i][c];

    // Resolve each element vertex's index into the bundle's deformable dof
    // list. MORreplaceKPCA populates the map at scene build: Rigidify cross-
    // parent gets the SubsetMultiMapping-derived map (rigid verts -> -1),
    // single-mstate gets identity. There is no direct-index fallback —
    // bundles built before the map was plumbed must be rebuilt.
    std::vector<long> defIdx(V);
    for (unsigned i = 0; i < V; ++i)
    {
        if (static_cast<long>(indexList[i]) >= m_indexMap.size())
            throw std::runtime_error(
                "HyperReducedHelperKPCA: vertex index " +
                std::to_string(indexList[i]) +
                " exceeds indexMap size " +
                std::to_string(m_indexMap.size()) + ".");
        defIdx[i] = m_indexMap(indexList[i]);  // -1 marks a rigid vertex
    }

    // t_j = grad_j[elem_dofs] · Ginv_contrib  — (T,)
    Eigen::VectorXd t(T);
    for (unsigned j = 0; j < T; ++j)
    {
        double s = 0.0;
        for (unsigned i = 0; i < V; ++i)
        {
            if (defIdx[i] < 0) continue;
            const unsigned dof0 = 3 * static_cast<unsigned>(defIdx[i]);
            s += m_grad_all(dof0 + 0, j) * Ginv_contrib(3 * i + 0)
               + m_grad_all(dof0 + 1, j) * Ginv_contrib(3 * i + 1)
               + m_grad_all(dof0 + 2, j) * Ginv_contrib(3 * i + 2);
        }
        t(j) = s;
    }

    // Augmented Gie row: [Φ_t^T f_e ; α · t]. Rigid rows do NOT go through
    // G^{-1} — Φ_t is a constant decoder block, not a kernel-projected one.
    Eigen::VectorXd GieUnit(m_nbModes);
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
    GieUnit.segment(m_nbRigid, mDef).noalias() = m_projector->alpha() * t;
    return GieUnit;
}


template <class DataTypes>
void HyperReducedHelperKPCA::updateGie(
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

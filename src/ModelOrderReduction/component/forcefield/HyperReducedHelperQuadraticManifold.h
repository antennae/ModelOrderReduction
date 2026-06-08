/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*   Residual-kernel hyperreduction helper.                                    *
******************************************************************************/
#pragma once
#include <ModelOrderReduction/config.h>

#include <ModelOrderReduction/component/kernel/QuadraticManifoldProjector.h>

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

class SOFA_MODELORDERREDUCTION_API HyperReducedHelperQuadraticManifold
    : public virtual BaseObject
{
public:
    SOFA_CLASS(HyperReducedHelperQuadraticManifold, BaseObject);

    Data<bool>         d_prepareECSW;
    Data<std::string>  d_quadraticManifoldBundle;
    Data<unsigned int> d_nbTrainingSet;
    Data<unsigned int> d_periodSaveGIE;

    Data<bool>         d_performECSW;
    Data<std::string>  d_RIDPath;
    Data<std::string>  d_weightsPath;
    Data<sofa::type::vector<int>> d_indexMap;

    std::unique_ptr<sofa::component::kernel::QuadraticManifoldProjector> m_projector;
    unsigned int m_nbModes = 0;
    unsigned int m_nbDef = 0;
    unsigned int m_nbRigid = 0;
    Eigen::MatrixXd m_J;          // (3N, m)
    bool m_frameReady = false;
    Eigen::VectorXi m_indexMap;

    std::vector<std::vector<double>> Gie;
    Eigen::VectorXd weights;
    Eigen::VectorXi reducedIntegrationDomain;
    unsigned int m_RIDsize = 0;

    HyperReducedHelperQuadraticManifold();

    void initMOR(unsigned int nbElements, bool printLog = false);
    void prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& q);

    template <class DataTypes>
    Eigen::VectorXd projectOneElement(
        const std::vector<unsigned int>&              indexList,
        const std::vector<typename DataTypes::Deriv>& contrib) const;

    template <class DataTypes>
    void updateGie(const std::vector<unsigned int>&              indexList,
                   const std::vector<typename DataTypes::Deriv>& contrib,
                   unsigned int                                  numElem);

    void saveGieFile(unsigned int nbElements);
};


template <class DataTypes>
Eigen::VectorXd HyperReducedHelperQuadraticManifold::projectOneElement(
    const std::vector<unsigned int>&              indexList,
    const std::vector<typename DataTypes::Deriv>& contrib) const
{
    const unsigned V = static_cast<unsigned>(indexList.size());
    std::vector<long> defIdx(V);
    for (unsigned i = 0; i < V; ++i)
    {
        if (static_cast<long>(indexList[i]) >= m_indexMap.size())
            throw std::runtime_error(
                "HyperReducedHelperQuadraticManifold: vertex index " +
                std::to_string(indexList[i]) +
                " exceeds indexMap size " +
                std::to_string(m_indexMap.size()) + ".");
        defIdx[i] = m_indexMap(indexList[i]);
    }

    Eigen::VectorXd GieUnit(m_nbModes);
    for (unsigned k = 0; k < m_nbModes; ++k)
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
        GieUnit(k) = s;
    }
    return GieUnit;
}


template <class DataTypes>
void HyperReducedHelperQuadraticManifold::updateGie(
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

/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperQuadraticManifold.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/helper/logging/Messaging.h>

#include <fstream>
#include <sstream>

namespace modelorderreduction
{

using sofa::component::loader::MatrixLoader;
using sofa::component::kernel::QuadraticManifoldProjector;

HyperReducedHelperQuadraticManifold::HyperReducedHelperQuadraticManifold()
    : d_prepareECSW(initData(&d_prepareECSW, false, "prepareECSW",
          "Save data necessary for the construction of the reduced model"))
    , d_quadraticManifoldBundle(initData(&d_quadraticManifoldBundle, std::string("residual_kernel"),
          "quadraticManifoldBundle",
          "Path to quadratic-manifold bundle directory"))
    , d_nbTrainingSet(initData(&d_nbTrainingSet, unsigned(40), "nbTrainingSet",
          "When preparing the ECSW, size of the training set"))
    , d_periodSaveGIE(initData(&d_periodSaveGIE, unsigned(5), "periodSaveGIE",
          "When prepareECSW is true, the values of Gie are taken every periodSaveGIE timesteps."))
    , d_performECSW(initData(&d_performECSW, false, "performECSW",
          "Use the reduced model with the ECSW method"))
    , d_RIDPath(initData(&d_RIDPath, std::string("reducedIntegrationDomain.txt"), "RIDPath",
          "Path to the Reduced Integration domain when performing the ECSW method"))
    , d_weightsPath(initData(&d_weightsPath, std::string("weights.txt"), "weightsPath",
          "Path to the weights when performing ECSW"))
    , d_indexMap(initData(&d_indexMap, "indexMap",
          "Per-mstate-vertex map into the quadratic-manifold bundle slot."))
{
    static const std::string groupName{"HyperReduction"};
    d_prepareECSW.setGroup(groupName);
    d_quadraticManifoldBundle.setGroup(groupName);
    d_nbTrainingSet.setGroup(groupName);
    d_periodSaveGIE.setGroup(groupName);
    d_performECSW.setGroup(groupName);
    d_RIDPath.setGroup(groupName);
    d_weightsPath.setGroup(groupName);
    d_indexMap.setGroup(groupName);
}


void HyperReducedHelperQuadraticManifold::initMOR(unsigned nbElements, bool printLog)
{
    if (d_prepareECSW.getValue())
    {
        m_projector = std::make_unique<QuadraticManifoldProjector>();
        m_projector->loadFromBundle(d_quadraticManifoldBundle.getValue());
        m_nbModes = m_projector->nbModes();
        m_nbDef = m_nbModes;
        m_nbRigid = 0;

        if (printLog)
            msg_info("HyperReducedHelperQuadraticManifold")
                << "loaded quadratic-manifold bundle " << d_quadraticManifoldBundle.getValue()
                << "  3N=" << m_projector->nbDofs()
                << "  m=" << m_nbModes;

        Gie.assign(d_nbTrainingSet.getValue() * m_nbModes,
                   std::vector<double>(nbElements, 0.0));

        const auto& mapData = d_indexMap.getValue();
        if (mapData.empty())
            throw std::runtime_error(
                "HyperReducedHelperQuadraticManifold: indexMap Data field is empty.");
        m_indexMap.resize(mapData.size());
        for (std::size_t i = 0; i < mapData.size(); ++i)
            m_indexMap(static_cast<Eigen::Index>(i)) = mapData[i];
    }

    if (d_performECSW.getValue())
    {
        MatrixLoader<Eigen::VectorXd> wLoader;
        wLoader.m_printLog = printLog;
        wLoader.setFileName(d_weightsPath.getValue());
        wLoader.load();
        wLoader.getMatrix(weights);

        MatrixLoader<Eigen::VectorXi> rLoader;
        rLoader.m_printLog = printLog;
        rLoader.setFileName(d_RIDPath.getValue());
        rLoader.load();
        rLoader.getMatrix(reducedIntegrationDomain);

        m_RIDsize = reducedIntegrationDomain.rows();
        if (m_RIDsize == 0)
        {
            msg_warning("HyperReducedHelperQuadraticManifold")
                << "RID is empty! Integrating over all elements.";
            m_RIDsize = nbElements;
            reducedIntegrationDomain.resize(m_RIDsize);
            for (unsigned i = 0; i < m_RIDsize; ++i)
                reducedIntegrationDomain(i) = i;
        }
    }
    else
    {
        m_RIDsize = nbElements;
        reducedIntegrationDomain.resize(m_RIDsize);
        for (unsigned i = 0; i < m_RIDsize; ++i)
            reducedIntegrationDomain(i) = i;
    }
}


void HyperReducedHelperQuadraticManifold::prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& q)
{
    m_J = m_projector->J(q);
    m_frameReady = true;
}


void HyperReducedHelperQuadraticManifold::saveGieFile(unsigned nbElements)
{
    if (d_prepareECSW.getValue())
    {
        unsigned int numTest = int(this->getContext()->getTime()/this->getContext()->getDt());
        if (numTest%d_periodSaveGIE.getValue() == 0)
        {
            numTest = numTest/d_periodSaveGIE.getValue();
            if (numTest < d_nbTrainingSet.getValue()){
                std::stringstream gieFileNameSS;
                gieFileNameSS << this->name << "_Gie.txt";
                std::string gieFileName = gieFileNameSS.str();
                std::ofstream myfileGie(gieFileName, std::fstream::app);
                msg_info(this) << "Storing case number " << numTest+1
                               << " in " << gieFileName << " ...";
                for (unsigned int k=numTest*m_nbModes; k<(numTest+1)*m_nbModes;k++){
                    for (unsigned int l=0;l<nbElements;l++){
                        myfileGie << Gie[k][l] << " ";
                    }
                    myfileGie << std::endl;
                }
                myfileGie.close();
                msg_info(this) << "Storing Done";
            }
        }
    }
}

} // namespace modelorderreduction

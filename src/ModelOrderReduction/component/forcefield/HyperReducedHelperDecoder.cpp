/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperDecoder.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/helper/logging/Messaging.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace modelorderreduction
{

using sofa::component::loader::MatrixLoader;

HyperReducedHelperDecoder::HyperReducedHelperDecoder()
    : d_prepareECSW(initData(&d_prepareECSW, false, "prepareECSW",
          "Save data necessary for the construction of the reduced model"))
    , d_nbTrainingSet(initData(&d_nbTrainingSet, unsigned(40), "nbTrainingSet",
          "When preparing the ECSW, size of the training set"))
    , d_periodSaveGIE(initData(&d_periodSaveGIE, unsigned(5), "periodSaveGIE",
          "When prepareECSW is true, the values of Gie are taken every periodSaveGIE timesteps."))
    , d_performECSW(initData(&d_performECSW, false, "performECSW",
          "Use the reduced model with the ECSW method"))
    , d_RIDPath(initData(&d_RIDPath, std::string("reducedIntegrationDomain.txt"), "RIDPath",
          "Path to the Reduced Integration domain when performing the ECSW method"))
    , d_weightsPath(initData(&d_weightsPath, std::string("weights.txt"), "weightsPath",
          "Path to the weights when performing the ECSW method"))
    , d_indexMap(initData(&d_indexMap, "indexMap",
          "Per-mstate-vertex map into the bundle's deformable slot (-1 = rigid "
          "vertex, no decoder contribution). Required when the FF's mstate is "
          "larger than the bundle's deformable dim (Rigidify topology)."))
{
    static const std::string groupName{"HyperReduction"};
    d_prepareECSW.setGroup(groupName);
    d_nbTrainingSet.setGroup(groupName);
    d_periodSaveGIE.setGroup(groupName);
    d_performECSW.setGroup(groupName);
    d_RIDPath.setGroup(groupName);
    d_weightsPath.setGroup(groupName);
    d_indexMap.setGroup(groupName);
}


void HyperReducedHelperDecoder::initMOR(unsigned nbElements, bool printLog)
{
    if (d_prepareECSW.getValue())
    {
        m_projector = createProjector();
        m_projector->loadFromBundle(bundlePath());

        m_nbRigid = m_projector->nbRigid();
        m_nbDef   = m_projector->nbModes();
        m_nbModes = m_nbDef + m_nbRigid;
        m_PhiT    = m_projector->rigidModes();   // (3N, nbRigid); empty if 0

        if (printLog)
            msg_info("HyperReducedHelperDecoder")
                << "loaded bundle " << bundlePath()
                << "  3N=" << m_projector->nbDofs()
                << "  m_def=" << m_nbDef
                << "  nbRigid=" << m_nbRigid
                << "  m_total=" << m_nbModes;

        Gie.assign(d_nbTrainingSet.getValue() * m_nbModes,
                   std::vector<double>(nbElements, 0.0));

        const auto& mapData = d_indexMap.getValue();
        if (mapData.empty())
            throw std::runtime_error(
                "HyperReducedHelperDecoder: indexMap Data field is empty. The "
                "MOR scene wrapper should always populate it; reduced scenes "
                "built before this plumbing landed must be regenerated.");
        m_indexMap.resize(mapData.size());
        for (std::size_t i = 0; i < mapData.size(); ++i)
            m_indexMap(static_cast<Eigen::Index>(i)) = mapData[i];
        if (printLog)
            msg_info("HyperReducedHelperDecoder")
                << "indexMap set: " << mapData.size() << " mstate verts -> "
                << "bundle slots (-1 entries: rigid verts skipped).";
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
            msg_warning("HyperReducedHelperDecoder")
                << "RID is empty! Integrating over all the elements!";
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


void HyperReducedHelperDecoder::prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& q)
{
    m_J = m_projector->J(q);
    m_frameReady = true;
}


void HyperReducedHelperDecoder::saveGieFile(unsigned nbElements)
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
                std::ofstream myfileGie (gieFileName, std::fstream::app);
                msg_info(this) << "Storing case number " << numTest+1 << " in " << gieFileName << " ...";
                for (unsigned int k=numTest*m_nbModes; k<(numTest+1)*m_nbModes;k++){
                    for (unsigned int l=0;l<nbElements;l++){
                        myfileGie << Gie[k][l] << " ";
                    }
                    myfileGie << std::endl;
                }
                myfileGie.close();
                msg_info(this) << "Storing Done";
            }
            else
            {
                msg_info(this) << d_nbTrainingSet.getValue() << "were already stored. Learning phase completed.";
            }
        }
    }
}

} // namespace modelorderreduction

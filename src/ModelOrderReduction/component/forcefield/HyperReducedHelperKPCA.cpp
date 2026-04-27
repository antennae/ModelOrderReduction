/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperKPCA.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/helper/logging/Messaging.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace modelorderreduction
{

using sofa::component::loader::MatrixLoader;
using sofa::component::kernel::KernelProjector;
using sofa::component::kernel::RBFKernel;

HyperReducedHelperKPCA::HyperReducedHelperKPCA()
    : d_prepareECSW(initData(&d_prepareECSW, false, "prepareECSW",
          "Save data necessary for the construction of the reduced model"))
    , d_kernelBundle(initData(&d_kernelBundle, std::string("kpca"), "kernelBundle",
          "Path to the kPCA bundle directory (X0.txt, snapshots.txt, alpha.txt, kernel.txt)"))
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
{
    static const std::string groupName{"HyperReduction"};
    d_prepareECSW.setGroup(groupName);
    d_kernelBundle.setGroup(groupName);
    d_nbTrainingSet.setGroup(groupName);
    d_periodSaveGIE.setGroup(groupName);
    d_performECSW.setGroup(groupName);
    d_RIDPath.setGroup(groupName);
    d_weightsPath.setGroup(groupName);
}


void HyperReducedHelperKPCA::initMOR(unsigned nbElements, bool printLog)
{
    // Gie collection
    if (d_prepareECSW.getValue())
    {
        m_projector = sofa::component::kernel::loadKernelProjectorFromBundle(
            d_kernelBundle.getValue());
        m_nbModes = m_projector->nbModes();

        // σ² is the G^{-1} scalar (linear kernel leaves σ² = 1).
        if (auto* rbf = dynamic_cast<const RBFKernel*>(m_projector.get()))
            m_sigma2 = rbf->sigma() * rbf->sigma();
        else
            m_sigma2 = 1.0;

        if (printLog)
            msg_info("HyperReducedHelperKPCA")
                << "loaded bundle " << d_kernelBundle.getValue()
                << "  kernel=" << m_projector->kernelName()
                << "  3N=" << m_projector->nbDofs()
                << "  T=" << m_projector->nbSnapshots()
                << "  m=" << m_nbModes;

        Gie.assign(d_nbTrainingSet.getValue() * m_nbModes,
                   std::vector<double>(nbElements, 0.0));
    }

    // apply ECSW reduction
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
            msg_warning("HyperReducedHelperKPCA")
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


void HyperReducedHelperKPCA::prepareFrame(const Eigen::Ref<const Eigen::VectorXd>& u)
{
    // grad_u(u, snapshots) — shape (3N, T). Cached once per step.
    m_grad_all = m_projector->grad_u(u, m_projector->snapshots());
    m_frameReady = true;
}


void HyperReducedHelperKPCA::saveGieFile(unsigned nbElements)
{
    if (d_prepareECSW.getValue())
    {
        unsigned int numTest = int(this->getContext()->getTime()/this->getContext()->getDt());
        if (numTest%d_periodSaveGIE.getValue() == 0)       // A new value was taken
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

/******************************************************************************
*  HyperReducedHelperResidualKernel_dump — parity-test helper.
******************************************************************************/
#include <ModelOrderReduction/component/forcefield/HyperReducedHelperResidualKernel.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/defaulttype/VecTypes.h>

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using modelorderreduction::HyperReducedHelperResidualKernel;
using sofa::component::loader::MatrixLoader;

static Eigen::MatrixXd read_matrix_d(const std::string& path)
{
    MatrixLoader<Eigen::MatrixXd> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXd M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return M;
}

static Eigen::MatrixXi read_matrix_i(const std::string& path)
{
    MatrixLoader<Eigen::MatrixXi> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXi M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return M;
}

static void write_vector(const std::string& path, const Eigen::VectorXd& v)
{
    std::ofstream f(path);
    f << v.size() << " 1\n";
    f << std::scientific;
    f.precision(17);
    for (Eigen::Index i = 0; i < v.size(); ++i)
        f << v(i) << "\n";
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: " << argv[0]
                  << " <bundle_dir> <q_path> <indices_path> <contrib_path> <out_path>\n";
        return 2;
    }

    Eigen::MatrixXd q_mat = read_matrix_d(argv[2]);
    Eigen::MatrixXi idx_mat = read_matrix_i(argv[3]);
    Eigen::MatrixXd contr_mat = read_matrix_d(argv[4]);
    if (q_mat.cols() != 1) throw std::runtime_error("q must be (m, 1)");
    if (idx_mat.cols() != 1) throw std::runtime_error("indices must be (V, 1)");
    if (contr_mat.cols() != 3) throw std::runtime_error("contrib must be (V, 3)");
    if (idx_mat.rows() != contr_mat.rows())
        throw std::runtime_error("indices rows != contrib rows");

    HyperReducedHelperResidualKernel helper;
    helper.d_prepareECSW.setValue(true);
    helper.d_residualKernelBundle.setValue(argv[1]);
    helper.d_nbTrainingSet.setValue(1);
    helper.d_periodSaveGIE.setValue(1);

    sofa::type::vector<int> indexMap;
    const int maxIdx = idx_mat.maxCoeff();
    indexMap.resize(static_cast<std::size_t>(maxIdx + 1));
    for (int i = 0; i <= maxIdx; ++i)
        indexMap[static_cast<std::size_t>(i)] = i;
    helper.d_indexMap.setValue(indexMap);
    helper.initMOR(1u, false);

    const unsigned V = static_cast<unsigned>(idx_mat.rows());
    std::vector<unsigned int> indexList(V);
    for (unsigned i = 0; i < V; ++i)
        indexList[i] = static_cast<unsigned>(idx_mat(i, 0));

    using Vec3d = sofa::defaulttype::Vec3dTypes;
    std::vector<Vec3d::Deriv> contrib(V);
    for (unsigned i = 0; i < V; ++i)
        contrib[i] = Vec3d::Deriv(contr_mat(i, 0), contr_mat(i, 1), contr_mat(i, 2));

    helper.prepareFrame(q_mat.col(0));
    const Eigen::VectorXd GieUnit =
        helper.projectOneElement<Vec3d>(indexList, contrib);
    write_vector(argv[5], GieUnit);

    std::cout << "dumped residual-kernel GieUnit (m=" << GieUnit.size()
              << ") for bundle=" << argv[1] << "\n";
    return 0;
}

/******************************************************************************
*  ResidualKernelProjector_dump — standalone parity-test helper.
*
*  Usage:
*    ResidualKernelProjector_dump <bundle_dir> <q_path> <out_dir>
*                                 [<dq_path> <f_path>]
******************************************************************************/
#include <ModelOrderReduction/component/kernel/ResidualKernelProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

static void write_matrix(const std::string& path, const Eigen::MatrixXd& M)
{
    std::ofstream f(path);
    f << M.rows() << " " << M.cols() << "\n";
    f << std::scientific;
    f.precision(17);
    for (Eigen::Index i = 0; i < M.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            f << M(i, j) << (j + 1 < M.cols() ? ' ' : '\n');
    }
}

static Eigen::MatrixXd read_matrix(const std::string& path)
{
    sofa::component::loader::MatrixLoader<Eigen::MatrixXd> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXd M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to read: " + path);
    return M;
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 6)
    {
        std::cerr << "usage: " << argv[0]
                  << " <bundle_dir> <q_path> <out_dir> [<dq_path> <f_path>]\n";
        return 2;
    }

    const std::string bundle_dir = argv[1];
    const std::string q_path = argv[2];
    const std::string out_dir = argv[3];

    std::filesystem::create_directories(out_dir);

    sofa::component::kernel::ResidualKernelProjector proj;
    proj.loadFromBundle(bundle_dir);

    Eigen::MatrixXd q_mat = read_matrix(q_path);
    if (q_mat.cols() != 1 || static_cast<unsigned>(q_mat.rows()) != proj.nbModes())
    {
        std::cerr << "q must have shape (" << proj.nbModes() << ", 1); got ("
                  << q_mat.rows() << ", " << q_mat.cols() << ")\n";
        return 2;
    }
    const Eigen::VectorXd q = q_mat.col(0);

    write_matrix(out_dir + "/decode.txt", proj.decode(q));
    write_matrix(out_dir + "/J.txt", proj.J(q));

    if (argc == 6)
    {
        Eigen::MatrixXd dq_mat = read_matrix(argv[4]);
        Eigen::MatrixXd f_mat = read_matrix(argv[5]);
        if (dq_mat.cols() != 1 || static_cast<unsigned>(dq_mat.rows()) != proj.nbModes())
        {
            std::cerr << "dq must have shape (" << proj.nbModes() << ", 1)\n";
            return 2;
        }
        if (f_mat.cols() != 1 || static_cast<unsigned>(f_mat.rows()) != proj.nbDofs())
        {
            std::cerr << "f must have shape (" << proj.nbDofs() << ", 1)\n";
            return 2;
        }
        write_matrix(out_dir + "/applyJ.txt", proj.applyJ(q, dq_mat.col(0)));
        write_matrix(out_dir + "/project_force.txt", proj.project_force(q, f_mat.col(0)));
    }

    std::cout << "dumped projector="
              << sofa::component::kernel::ResidualKernelProjector::projectorName()
              << "  3N=" << proj.nbDofs()
              << "  m=" << proj.nbModes()
              << "  r=" << proj.nbResidualModes()
              << "  T=" << proj.nbTrain() << "\n";
    return 0;
}

/******************************************************************************
*            Model Order Reduction plugin for SOFA                            *
*                         AE-decoder Galerkin projector                       *
******************************************************************************/
// Include system + std-library headers BEFORE libtorch — libtorch's bundled
// headers can conflict with libstdc++ <filesystem> / iterator_traits on
// gcc 12 + cxx11-ABI when the order is reversed.
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>

#include <torch/script.h>

#include <ModelOrderReduction/component/kernel/AEProjector.h>
#include <ModelOrderReduction/component/loader/MatrixLoader.inl>

#include <sofa/helper/logging/Messaging.h>

namespace sofa::component::kernel {

namespace {

Eigen::MatrixXd load_matrix(const std::string& path)
{
    sofa::component::loader::MatrixLoader<Eigen::MatrixXd> ld;
    ld.setFileName(path);
    ld.load();
    Eigen::MatrixXd M;
    ld.getMatrix(M);
    if (M.rows() == 0 || M.cols() == 0)
        throw std::runtime_error("failed to load (missing or empty): " + path);
    return M;
}

// SiLU: x * sigmoid(x); SiLU': sigmoid(x) * (1 + x * (1 - sigmoid(x))).
inline Eigen::ArrayXd silu(const Eigen::ArrayXd& x)
{
    return x / (1.0 + (-x).exp());
}
inline Eigen::ArrayXd silu_prime(const Eigen::ArrayXd& x)
{
    const Eigen::ArrayXd s = 1.0 / (1.0 + (-x).exp());
    return s * (1.0 + x * (1.0 - s));
}

// Convert a torch::Tensor (CPU, 2D float) to an Eigen::MatrixXd.
Eigen::MatrixXd tensor_to_eigen_matrix(const torch::Tensor& t)
{
    auto tc = t.detach().to(torch::kFloat64).contiguous().cpu();
    const auto sz = tc.sizes();
    if (sz.size() != 2)
        throw std::runtime_error("expected 2D tensor for weight matrix");
    Eigen::MatrixXd M(sz[0], sz[1]);
    std::memcpy(M.data(), tc.data_ptr<double>(), M.size() * sizeof(double));
    // libtorch stores row-major; Eigen default is column-major → need transpose.
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        tc.data_ptr<double>(), sz[0], sz[1]);
}

Eigen::VectorXd tensor_to_eigen_vector(const torch::Tensor& t)
{
    auto tc = t.detach().to(torch::kFloat64).contiguous().cpu();
    const auto sz = tc.sizes();
    if (sz.size() != 1)
        throw std::runtime_error("expected 1D tensor for bias vector");
    Eigen::VectorXd v(sz[0]);
    std::memcpy(v.data(), tc.data_ptr<double>(), v.size() * sizeof(double));
    return v;
}

} // anonymous namespace

// Hide the libtorch encoder behind a PIMPL so users of AEProjector.h don't
// need <torch/script.h> in their includes.
struct AEProjector::EncoderImpl
{
    torch::jit::script::Module mod;
};

AEProjector::AEProjector() = default;
AEProjector::~AEProjector() = default;

const std::string& AEProjector::projectorName()
{
    static const std::string n = "ae";
    return n;
}

void AEProjector::loadFromBundle(const std::string& bundle_dir)
{
    namespace fs = std::filesystem;
    const fs::path root(bundle_dir);

    // 1) Reference state + per-DOF unnormaliser.
    Eigen::MatrixXd X0_mat       = load_matrix((root / "X0.txt").string());
    Eigen::MatrixXd col_std_mat  = load_matrix((root / "col_std.txt").string());
    if (X0_mat.cols() != 1)
        throw std::runtime_error("X0.txt must have 1 column");
    if (col_std_mat.cols() != 1 || col_std_mat.rows() != X0_mat.rows())
        throw std::runtime_error("col_std.txt must be (3N, 1) matching X0");

    m_X0      = X0_mat.col(0);
    m_col_std = col_std_mat.col(0);
    m_nbDofs  = static_cast<unsigned>(m_X0.size());

    // 2) Optional rigid translation columns.
    const auto rigid_path = (root / "rigid_modes.txt").string();
    if (fs::exists(rigid_path))
    {
        m_rigidModes = load_matrix(rigid_path);
        if (static_cast<unsigned>(m_rigidModes.rows()) != m_nbDofs)
            throw std::runtime_error("rigid_modes rows != X0 length");
        if (m_rigidModes.cols() < 1 || m_rigidModes.cols() > 3)
            throw std::runtime_error("rigid_modes must have 1-3 columns");
    }

    // 3) Decoder weights — load TorchScript module and extract Linear params
    //    into Eigen. The module is a Sequential of [Linear, SiLU, ..., Linear]
    //    wrapped to apply col_std unnormalisation as the final step. We strip
    //    that wrapper here by extracting the raw decoder parameters; col_std
    //    is applied separately in C++ so J() can be computed analytically.
    const auto decoder_path = (root / "decoder.ts.pt").string();
    if (!fs::exists(decoder_path))
        throw std::runtime_error("decoder.ts.pt not found in bundle: " + bundle_dir);

    torch::jit::script::Module decoder_mod;
    try {
        decoder_mod = torch::jit::load(decoder_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error("failed to load TorchScript decoder: " + std::string(e.what()));
    }
    decoder_mod.eval();

    // Walk named_parameters, expect pairs of (..weight, ..bias) per Linear layer
    // in declaration order. The Sequential's children are 0, 2, 4, ... (Linear)
    // and 1, 3, 5, ... (SiLU, no params). torch::jit lists params in registration
    // order so we can collect them in pairs.
    std::vector<std::pair<std::string, torch::Tensor>> params;
    for (const auto& p : decoder_mod.named_parameters(/*recurse=*/true))
        params.emplace_back(p.name, p.value);

    if (params.size() < 2 || params.size() % 2 != 0)
        throw std::runtime_error("decoder must have an even number of params (W,b per layer); got "
                                 + std::to_string(params.size()));

    m_weights.clear();
    m_biases.clear();
    for (size_t i = 0; i < params.size(); i += 2)
    {
        const auto& wname = params[i].first;
        const auto& bname = params[i + 1].first;
        if (wname.find("weight") == std::string::npos || bname.find("bias") == std::string::npos)
            throw std::runtime_error("expected (weight, bias) pair, got '" + wname + "', '" + bname + "'");
        m_weights.push_back(tensor_to_eigen_matrix(params[i].second));
        m_biases.push_back(tensor_to_eigen_vector(params[i + 1].second));
    }
    // Sanity-check shapes form a chain: W_i has rows = b_i.size(), cols = W_{i-1}.rows() (or m for i=0).
    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        if (m_weights[i].rows() != m_biases[i].size())
            throw std::runtime_error("layer " + std::to_string(i) + ": W rows != b size");
        if (i > 0 && m_weights[i].cols() != m_weights[i - 1].rows())
            throw std::runtime_error("layer " + std::to_string(i) + ": dim mismatch with previous");
    }
    m_nbModes = static_cast<unsigned>(m_weights.front().cols());
    if (static_cast<unsigned>(m_weights.back().rows()) != m_nbDofs)
        throw std::runtime_error("decoder output dim != 3N from X0");

    m_activation = "silu";  // v1 only

    // 4) Optional encoder.
    const auto encoder_path = (root / "encoder.ts.pt").string();
    if (fs::exists(encoder_path))
    {
        m_encoder = std::make_unique<EncoderImpl>();
        try {
            m_encoder->mod = torch::jit::load(encoder_path);
            m_encoder->mod.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("failed to load TorchScript encoder: " + std::string(e.what()));
        }
    }
}

// ---- forward + analytical Jacobian ----

namespace {

// Forward pass returning per-layer pre-activations z_i (used by the Jacobian
// chain rule). Output of last layer is z_{n-1} (no activation applied to it).
struct ForwardCache {
    std::vector<Eigen::VectorXd> z;   // pre-activations, one per layer
};

ForwardCache forward_with_cache(const std::vector<Eigen::MatrixXd>& W,
                                const std::vector<Eigen::VectorXd>& b,
                                const Eigen::VectorXd& q)
{
    ForwardCache c;
    c.z.reserve(W.size());
    Eigen::VectorXd x = q;
    for (size_t i = 0; i < W.size(); ++i)
    {
        Eigen::VectorXd z = W[i] * x + b[i];
        c.z.push_back(z);
        if (i + 1 < W.size())
            x = silu(z.array()).matrix();
        else
            x = z;
    }
    return c;
}

} // anon

AEProjector::VectorXd AEProjector::decode(const Eigen::Ref<const VectorXd>& q) const
{
    if (static_cast<unsigned>(q.size()) != m_nbModes)
        throw std::runtime_error("decode: q.size != nbModes");
    const auto c = forward_with_cache(m_weights, m_biases, q);
    return m_col_std.cwiseProduct(c.z.back());     // u_disp = col_std ⊙ g(q)
}

AEProjector::MatrixXd AEProjector::J(const Eigen::Ref<const VectorXd>& q) const
{
    if (static_cast<unsigned>(q.size()) != m_nbModes)
        throw std::runtime_error("J: q.size != nbModes");

    // J = diag(col_std) · W_n · D_{n-1} · W_{n-1} · D_{n-2} · ... · W_1
    // where D_i = diag(σ'(z_i)).  Build right-to-left so the running dim is m
    // until the final left-multiply by diag(col_std).
    const auto c = forward_with_cache(m_weights, m_biases, q);
    Eigen::MatrixXd Jcur = m_weights.front();          // (h_1, m)
    for (size_t i = 1; i < m_weights.size(); ++i)
    {
        const Eigen::VectorXd dprime = silu_prime(c.z[i - 1].array()).matrix();  // (h_{i-1},)
        // D_{i-1} · Jcur — row-scale Jcur by dprime
        Jcur.array().colwise() *= dprime.array();
        Jcur = m_weights[i] * Jcur;                    // (h_i, m)
    }
    // Final: diag(col_std) · Jcur
    Jcur.array().colwise() *= m_col_std.array();
    return Jcur;                                       // (3N, m)
}

AEProjector::VectorXd AEProjector::applyJ(const Eigen::Ref<const VectorXd>& q,
                                          const Eigen::Ref<const VectorXd>& dq) const
{
    if (static_cast<unsigned>(dq.size()) != m_nbModes)
        throw std::runtime_error("applyJ: dq.size != nbModes");
    // For v1 just do J · dq via the cached path. Stage F can replace this with
    // a direct JVP that avoids the (3N, m) materialisation.
    return J(q) * dq;
}

AEProjector::VectorXd AEProjector::project_force(const Eigen::Ref<const VectorXd>& q,
                                                 const Eigen::Ref<const VectorXd>& f) const
{
    if (static_cast<unsigned>(f.size()) != m_nbDofs)
        throw std::runtime_error("project_force: f.size != nbDofs");
    return J(q).transpose() * f;
}

AEProjector::VectorXd AEProjector::encode(const Eigen::Ref<const VectorXd>& u_disp) const
{
    if (!m_encoder)
        throw std::runtime_error("encode: no encoder.ts.pt in bundle");
    if (static_cast<unsigned>(u_disp.size()) != m_nbDofs)
        throw std::runtime_error("encode: u_disp.size != nbDofs");

    // The encoder TorchScript wraps col_std pre-normalisation, so feed u_disp directly.
    auto t_in = torch::from_blob(const_cast<double*>(u_disp.data()),
                                 {static_cast<long>(u_disp.size())},
                                 torch::TensorOptions().dtype(torch::kFloat64))
                    .clone()
                    .to(torch::kFloat32);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(t_in);
    const auto out = m_encoder->mod.forward(inputs).toTensor().to(torch::kFloat64).contiguous().cpu();

    Eigen::VectorXd q(out.size(0));
    std::memcpy(q.data(), out.data_ptr<double>(), q.size() * sizeof(double));
    return q;
}

} // namespace sofa::component::kernel

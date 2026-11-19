// Shared pieces of the model library: the dense matrix, the loss used to
// compare families, JSON dispatch, and the blend model.

#include "neurograph/models/model.hpp"
#include "detail.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace neurograph::models {

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

Matrix Matrix::from_rows(const std::vector<std::vector<double>>& rows_in) {
    Matrix m;
    m.rows = rows_in.size();
    for (const auto& r : rows_in) m.cols = std::max(m.cols, r.size());
    m.data.assign(m.rows * m.cols, 0.0);
    for (std::size_t i = 0; i < m.rows; ++i) {
        std::copy(rows_in[i].begin(), rows_in[i].end(), m.row(i));
    }
    return m;
}

// ---------------------------------------------------------------------------
// evaluate_loss
// ---------------------------------------------------------------------------

double evaluate_loss(const Model& model,
                     const Matrix& X,
                     const std::vector<int>&    y_cls,
                     const std::vector<double>& y_reg) {
    if (X.rows == 0) return std::numeric_limits<double>::infinity();
    const bool cls = model.task() == Task::Classification;
    std::vector<double> out(static_cast<std::size_t>(std::max(1, model.n_outputs())));
    double total = 0.0;
    for (std::size_t i = 0; i < X.rows; ++i) {
        model.predict(X.row(i), out.data());
        if (cls) {
            const int y = y_cls[i];
            const double p = (y >= 0 && y < static_cast<int>(out.size())) ? out[y] : 0.0;
            total += -std::log(std::max(p, detail::kProbFloor));
        } else {
            const double d = out[0] - y_reg[i];
            total += d * d;
        }
    }
    return total / static_cast<double>(X.rows);
}

// ---------------------------------------------------------------------------
// Blend
// ---------------------------------------------------------------------------

namespace {

class BlendModel final : public Model {
public:
    BlendModel(std::vector<std::shared_ptr<const Model>> members,
               std::vector<double> weights)
        : members_(std::move(members)), weights_(std::move(weights)) {
        if (members_.empty() || members_.size() != weights_.size()) {
            throw std::runtime_error("blend: members and weights must match and be non-empty");
        }
        double sum = 0.0;
        for (double w : weights_) sum += w;
        if (!(sum > 0.0)) throw std::runtime_error("blend: weights must sum to a positive value");
        for (double& w : weights_) w /= sum;
        for (const auto& m : members_) {
            if (!m) throw std::runtime_error("blend: null member");
            if (m->task() != members_.front()->task()
                || m->n_features() != members_.front()->n_features()
                || m->n_outputs()  != members_.front()->n_outputs()) {
                throw std::runtime_error("blend: members disagree on task or shape");
            }
        }
    }

    const char* kind() const noexcept override { return "blend_v1"; }
    Task task() const noexcept override { return members_.front()->task(); }
    int  n_features() const noexcept override { return members_.front()->n_features(); }
    int  n_outputs()  const noexcept override { return members_.front()->n_outputs(); }

    void predict(const double* x, double* out) const override {
        const int k = n_outputs();
        std::vector<double> tmp(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) out[i] = 0.0;
        for (std::size_t m = 0; m < members_.size(); ++m) {
            members_[m]->predict(x, tmp.data());
            for (int i = 0; i < k; ++i) out[i] += weights_[m] * tmp[i];
        }
    }

    nlohmann::json to_json() const override {
        nlohmann::json j;
        j["kind"]    = kind();
        j["weights"] = weights_;
        j["members"] = nlohmann::json::array();
        for (const auto& m : members_) j["members"].push_back(m->to_json());
        return j;
    }

    std::string describe() const override {
        std::ostringstream os;
        os << "Blend of ";
        for (std::size_t m = 0; m < members_.size(); ++m) {
            if (m) os << " and ";
            os << static_cast<int>(std::lround(weights_[m] * 100.0)) << "% "
               << members_[m]->describe();
        }
        return os.str();
    }

    std::size_t parameter_count() const noexcept override {
        std::size_t n = 0;
        for (const auto& m : members_) n += m->parameter_count();
        return n;
    }

private:
    std::vector<std::shared_ptr<const Model>> members_;
    std::vector<double>                       weights_;
};

} // namespace

std::unique_ptr<Model> make_blend(std::vector<std::shared_ptr<const Model>> members,
                                  std::vector<double> weights) {
    return std::make_unique<BlendModel>(std::move(members), std::move(weights));
}

std::unique_ptr<Model> detail::blend_from_json(const nlohmann::json& j) {
    std::vector<std::shared_ptr<const Model>> members;
    for (const auto& mj : j.at("members")) {
        members.push_back(std::shared_ptr<const Model>(model_from_json(mj)));
    }
    return make_blend(std::move(members), j.at("weights").get<std::vector<double>>());
}

// ---------------------------------------------------------------------------
// JSON dispatch
// ---------------------------------------------------------------------------

std::unique_ptr<Model> model_from_json(const nlohmann::json& j) {
    if (!j.is_object() || !j.contains("kind") || !j.at("kind").is_string()) {
        throw std::runtime_error("model: missing 'kind'");
    }
    const std::string kind = j.at("kind").get<std::string>();
    try {
        if (kind == "mlp_v1" || kind == "linear_v1") return detail::mlp_from_json(j);
        if (kind == "gbt_v1")                         return detail::gbt_from_json(j);
        if (kind == "blend_v1")                       return detail::blend_from_json(j);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("model '" + kind + "': " + e.what());
    }
    throw std::runtime_error("model: unknown kind '" + kind + "'");
}

} // namespace neurograph::models

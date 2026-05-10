#include "spectral_bandit/bandit.hpp"

#include <Eigen/Cholesky>

#include <stdexcept>

namespace spectral_bandit {

LinearUCB::LinearUCB(int dim, double alpha, double lam)
    : dim_(dim), alpha_(alpha), lam_(lam), V_(Eigen::MatrixXd::Identity(dim, dim) * lam), b_(Eigen::VectorXd::Zero(dim)) {
  if (dim <= 0) {
    throw std::runtime_error("LinearUCB: dim must be >= 1");
  }
  if (alpha < 0.0) {
    throw std::runtime_error("LinearUCB: alpha must be >= 0");
  }
  if (lam <= 0.0) {
    throw std::runtime_error("LinearUCB: lam must be > 0");
  }
}

Eigen::VectorXd LinearUCB::theta() const { return V_.ldlt().solve(b_); }

BanditScores LinearUCB::scores(const Eigen::MatrixXd& contexts) const {
  if (contexts.cols() != dim_) {
    throw std::runtime_error("LinearUCB::scores: contexts must have shape (n_arms, dim)");
  }

  const Eigen::MatrixXd Vinv = V_.ldlt().solve(Eigen::MatrixXd::Identity(dim_, dim_));
  const Eigen::VectorXd th = Vinv * b_;

  Eigen::VectorXd mu = contexts * th;
  Eigen::VectorXd sigma(contexts.rows());
  for (Eigen::Index i = 0; i < contexts.rows(); ++i) {
    const Eigen::VectorXd x = contexts.row(i).transpose();
    sigma(i) = std::sqrt(std::max(0.0, x.dot(Vinv * x)));
  }

  Eigen::VectorXd ucb = mu + alpha_ * sigma;
  return BanditScores{.mu = mu, .sigma = sigma, .ucb = ucb};
}

std::pair<int, BanditScores> LinearUCB::select(const Eigen::MatrixXd& contexts,
                                                const std::optional<Eigen::VectorXd>& travel_cost,
                                                double beta) const {
  BanditScores s = scores(contexts);
  Eigen::VectorXd val = s.ucb;

  if (travel_cost.has_value()) {
    if (travel_cost->size() != contexts.rows()) {
      throw std::runtime_error("LinearUCB::select: travel_cost must have one value per arm");
    }
    val = val - beta * (*travel_cost);
  }

  Eigen::Index argmax = 0;
  val.maxCoeff(&argmax);
  return {static_cast<int>(argmax), BanditScores{.mu = s.mu, .sigma = s.sigma, .ucb = val}};
}

void LinearUCB::update(const Eigen::VectorXd& context, double reward) {
  if (context.size() != dim_) {
    throw std::runtime_error("LinearUCB::update: context must have shape (dim)");
  }
  V_ += context * context.transpose();
  b_ += reward * context;
}

}  // namespace spectral_bandit

#include "spectral_bandit/simulator.hpp"

#include "spectral_bandit/heat.hpp"
#include "spectral_bandit/pathfinding.hpp"
#include "spectral_bandit/spectral.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace spectral_bandit {

SpectralBanditSimulator::SpectralBanditSimulator(const SimulatorConfig& config, Mesh mesh, SparseMatrix laplacian,
                                                 Eigen::VectorXd mass_diag, SparseMatrix mass_mat,
                                                 PatchificationResult patchification, Eigen::MatrixXd vertex_embeddings,
                                                 Eigen::MatrixXd patch_contexts, AdjacencyList adjacency,
                                                 Eigen::VectorXd hidden_patch_probs, std::mt19937_64 rng)
    : config_(config),
      mesh_(std::move(mesh)),
      laplacian_(std::move(laplacian)),
      mass_diag_(std::move(mass_diag)),
      mass_mat_(std::move(mass_mat)),
      patchification_(std::move(patchification)),
      vertex_embeddings_(std::move(vertex_embeddings)),
      patch_contexts_(std::move(patch_contexts)),
      adjacency_(std::move(adjacency)),
      hidden_patch_probs_(std::move(hidden_patch_probs)),
      bandit_(static_cast<int>(patch_contexts_.cols()), config_.alpha, config_.lam),
      rng_(std::move(rng)) {
  if (patchification_.center_vertices.size() <= 0) {
    throw std::runtime_error("SpectralBanditSimulator: no patch center vertices");
  }
  current_vertex_ = patchification_.center_vertices(0);
}

SpectralBanditSimulator SpectralBanditSimulator::from_config(const SimulatorConfig& config) {
  std::mt19937_64 rng(static_cast<std::uint64_t>(config.seed));

  Mesh mesh = load_obj(config.mesh_path);
  auto [mass_diag, mass_mat] = mass_matrix(mesh);
  SparseMatrix laplacian = cotangent_laplacian(mesh);

  HeatMethodGeodesics heat(mesh, laplacian, mass_diag);
  PatchificationResult patchification = geodesic_patchify(mesh, heat, mass_diag, config.n_patches, config.seed);

  SpectralEmbeddingResult spec = spectral_embedding(laplacian, mass_mat, config.spectral_dim, true);

  Eigen::MatrixXd patch_contexts =
      patch_embeddings(spec.vertex_embeddings, patchification.vertex_labels, mass_diag, config.n_patches);

  if (config.add_bias_feature) {
    Eigen::MatrixXd with_bias(patch_contexts.rows(), patch_contexts.cols() + 1);
    with_bias.leftCols(patch_contexts.cols()) = patch_contexts;
    with_bias.col(with_bias.cols() - 1).setOnes();
    patch_contexts = with_bias;
  }

  AdjacencyList adjacency = build_graph_adjacency(mesh);

  Eigen::MatrixXd center_positions(config.n_patches, 3);
  for (int i = 0; i < config.n_patches; ++i) {
    center_positions.row(i) = mesh.vertices.row(patchification.center_vertices(i));
  }

  Eigen::VectorXd hidden =
      make_hidden_probabilities(patch_contexts, adjacency, patchification.center_vertices, center_positions, config, rng);

  return SpectralBanditSimulator(config, std::move(mesh), std::move(laplacian), std::move(mass_diag), std::move(mass_mat),
                                 std::move(patchification), std::move(spec.vertex_embeddings),
                                 std::move(patch_contexts), std::move(adjacency), std::move(hidden), std::move(rng));
}

Eigen::VectorXd SpectralBanditSimulator::make_hidden_probabilities(const Eigen::MatrixXd& patch_contexts,
                                                                   const AdjacencyList& adjacency,
                                                                   const Eigen::VectorXi& center_vertices,
                                                                   const Eigen::MatrixXd& center_positions,
                                                                   const SimulatorConfig& config,
                                                                   std::mt19937_64& rng) {
  const int n_patches = static_cast<int>(patch_contexts.rows());

  std::normal_distribution<double> normal(0.0, 0.7);
  Eigen::VectorXd theta_bg(patch_contexts.cols());
  for (Eigen::Index i = 0; i < theta_bg.size(); ++i) {
    theta_bg(i) = normal(rng);
  }

  Eigen::VectorXd bg_raw = patch_contexts * theta_bg;
  const double mean = bg_raw.mean();
  const double var = (bg_raw.array() - mean).square().mean();
  const double stddev = std::sqrt(std::max(var, 1e-12));
  bg_raw = (bg_raw.array() - mean) / (stddev + 1e-12);
  Eigen::VectorXd background = config.gold_background_scale * bg_raw;

  int hotspot_count = config.gold_hotspot_count;
  if (hotspot_count <= 0) {
    hotspot_count = std::max(1, std::min(5, n_patches / 70));
  }

  std::uniform_int_distribution<int> dis_patch(0, n_patches - 1);
  std::vector<int> seeds;
  seeds.reserve(static_cast<std::size_t>(hotspot_count));
  seeds.push_back(dis_patch(rng));

  Eigen::VectorXd d_min(n_patches);
  for (int i = 0; i < n_patches; ++i) {
    d_min(i) = (center_positions.row(i) - center_positions.row(seeds[0])).norm();
  }

  while (static_cast<int>(seeds.size()) < hotspot_count) {
    Eigen::Index arg = 0;
    d_min.maxCoeff(&arg);
    int c_new = static_cast<int>(arg);

    if (std::find(seeds.begin(), seeds.end(), c_new) != seeds.end()) {
      std::vector<int> order(static_cast<std::size_t>(n_patches));
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&d_min](int a, int b) { return d_min(a) > d_min(b); });
      for (int c : order) {
        if (std::find(seeds.begin(), seeds.end(), c) == seeds.end()) {
          c_new = c;
          break;
        }
      }
    }

    seeds.push_back(c_new);

    for (int i = 0; i < n_patches; ++i) {
      const double d = (center_positions.row(i) - center_positions.row(c_new)).norm();
      if (d < d_min(i)) {
        d_min(i) = d;
      }
    }
  }

  std::uniform_real_distribution<double> jitter(-config.gold_hotspot_sigma_jitter, config.gold_hotspot_sigma_jitter);
  std::uniform_real_distribution<double> amp_dis(config.gold_hotspot_amp_min, config.gold_hotspot_amp_max);

  Eigen::VectorXd hotspot_field = Eigen::VectorXd::Zero(n_patches);
  for (int seed_patch : seeds) {
    const int source_vertex = center_vertices(seed_patch);
    Eigen::VectorXd dist_vertices = dijkstra_distances(adjacency, source_vertex);

    Eigen::VectorXd dist_patches(n_patches);
    for (int i = 0; i < n_patches; ++i) {
      dist_patches(i) = dist_vertices(center_vertices(i));
    }

    double sigma = config.gold_hotspot_sigma + jitter(rng);
    sigma = std::max(0.05, sigma);
    const double amp = amp_dis(rng);

    hotspot_field.array() += amp * (-(dist_patches.array().square()) / (2.0 * sigma * sigma)).exp();
  }

  hotspot_field = hotspot_field.array().max(0.0).pow(1.25);

  std::normal_distribution<double> small_noise(0.0, 0.003);
  Eigen::VectorXd probs(n_patches);
  for (int i = 0; i < n_patches; ++i) {
    const double noise = small_noise(rng);
    double p = config.gold_base_prob + background(i) + hotspot_field(i) + noise;
    p = std::clamp(p, 0.001, config.gold_prob_max);
    probs(i) = p;
  }

  return probs;
}

int SpectralBanditSimulator::oracle_best_patch() const {
  Eigen::Index argmax = 0;
  hidden_patch_probs_.maxCoeff(&argmax);
  return static_cast<int>(argmax);
}

int SpectralBanditSimulator::cumulative_reward() const {
  int total = 0;
  for (const auto& s : history_) {
    total += s.reward;
  }
  return total;
}

Eigen::VectorXd SpectralBanditSimulator::travel_cost_to_patches() const {
  Eigen::VectorXd dist_to_vertices = dijkstra_distances(adjacency_, current_vertex_);
  Eigen::VectorXd cost(n_patches());
  for (int i = 0; i < n_patches(); ++i) {
    cost(i) = dist_to_vertices(patchification_.center_vertices(i));
  }
  return cost;
}

ScoreSnapshot SpectralBanditSimulator::current_scores() const {
  BanditScores raw = bandit_.scores(patch_contexts_);
  Eigen::VectorXd travel = travel_cost_to_patches();

  Eigen::VectorXd decision = raw.ucb;
  if (config_.beta > 0.0) {
    decision = decision - config_.beta * travel;
  }

  return ScoreSnapshot{.raw = raw, .travel_cost = travel, .decision = decision};
}

std::tuple<int, BanditScores, Eigen::VectorXd, Eigen::VectorXd> SpectralBanditSimulator::recommend_patch() const {
  ScoreSnapshot snap = current_scores();
  Eigen::Index argmax = 0;
  snap.decision.maxCoeff(&argmax);
  BanditScores scored{.mu = snap.raw.mu, .sigma = snap.raw.sigma, .ucb = snap.decision};
  return {static_cast<int>(argmax), scored, snap.travel_cost, snap.decision};
}

StepResult SpectralBanditSimulator::step(std::optional<int> target_patch) {
  ScoreSnapshot snap = current_scores();

  int patch = target_patch.has_value() ? *target_patch : 0;
  if (!target_patch.has_value()) {
    Eigen::Index argmax = 0;
    snap.decision.maxCoeff(&argmax);
    patch = static_cast<int>(argmax);
  }

  if (patch < 0 || patch >= n_patches()) {
    throw std::out_of_range("SpectralBanditSimulator::step: invalid patch index");
  }

  const int target_vertex = patchification_.center_vertices(patch);
  auto [path, travel_cost] = dijkstra_shortest_path(adjacency_, current_vertex_, target_vertex);

  if (path.size() == 0 || !std::isfinite(travel_cost)) {
    throw std::runtime_error("SpectralBanditSimulator::step: no valid path found");
  }

  current_vertex_ = target_vertex;

  const double p = hidden_patch_probs_(patch);
  std::bernoulli_distribution bernoulli(std::clamp(p, 0.0, 1.0));
  const int reward = bernoulli(rng_) ? 1 : 0;

  bandit_.update(patch_contexts_.row(patch).transpose(), static_cast<double>(reward));

  t_ += 1;
  StepResult out;
  out.t = t_;
  out.chosen_patch = patch;
  out.reward = reward;
  out.path = path;
  out.travel_cost = travel_cost;
  out.predicted_mu = snap.raw.mu(patch);
  out.predicted_sigma = snap.raw.sigma(patch);
  out.decision_score = snap.decision(patch);
  out.oracle_prob = p;

  history_.push_back(out);
  return out;
}

std::vector<StepResult> SpectralBanditSimulator::run(int steps) {
  if (steps < 0) {
    throw std::runtime_error("SpectralBanditSimulator::run: steps must be >= 0");
  }
  std::vector<StepResult> out;
  out.reserve(static_cast<std::size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    out.push_back(step(std::nullopt));
  }
  return out;
}

std::map<std::string, Eigen::VectorXd> SpectralBanditSimulator::heatmap_values() const {
  ScoreSnapshot snap = current_scores();
  return {
      {"exploit", snap.raw.mu},
      {"explore", snap.raw.sigma},
      {"ucb", snap.decision},
      {"cost", snap.travel_cost},
      {"oracle", hidden_patch_probs_},
  };
}

}  // namespace spectral_bandit

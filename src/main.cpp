#include "spectral_bandit/simulator.hpp"
#include "spectral_bandit/viewer.hpp"

#include <Eigen/Core>

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sb = spectral_bandit;

namespace {

struct Args {
  sb::SimulatorConfig config{};
  int steps = 30;
  std::optional<int> manual_patch;
  int steps_before_render = 20;
  std::string out_dir = "outputs";
  std::string player_mesh = "player/player_texture.obj";
};

void usage() {
  std::cout << "Usage:\n"
            << "  spectral_bandit_cpp simulate [options]\n"
            << "  spectral_bandit_cpp render [options]\n"
            << "  spectral_bandit_cpp viewer [options]\n"
            << "  spectral_bandit_cpp game [options]\n"
            << "  spectral_bandit_cpp game3d [options]\n\n"
            << "Common options:\n"
            << "  --mesh <path>\n"
            << "  --patches <int>\n"
            << "  --dim <int>\n"
            << "  --alpha <float>\n"
            << "  --lam <float>\n"
            << "  --beta <float>\n"
            << "  --seed <int>\n"
            << "  --no-bias\n\n"
            << "simulate options:\n"
            << "  --steps <int>\n"
            << "  --manual-patch <int>\n\n"
            << "render options:\n"
            << "  --steps-before-render <int>\n"
            << "  --out-dir <path>\n\n"
            << "game3d options:\n"
            << "  --player-mesh <path>\n";
}

std::string require_value(const std::vector<std::string>& argv, std::size_t& i, const std::string& opt) {
  if (i + 1 >= argv.size()) {
    throw std::runtime_error("Missing value for option " + opt);
  }
  i += 1;
  return argv[i];
}

Args parse_args(const std::string& cmd, const std::vector<std::string>& argv) {
  Args a;

  for (std::size_t i = 0; i < argv.size(); ++i) {
    const std::string& tok = argv[i];

    if (tok == "--mesh") {
      a.config.mesh_path = require_value(argv, i, tok);
    } else if (tok == "--patches") {
      a.config.n_patches = std::stoi(require_value(argv, i, tok));
    } else if (tok == "--dim") {
      a.config.spectral_dim = std::stoi(require_value(argv, i, tok));
    } else if (tok == "--alpha") {
      a.config.alpha = std::stod(require_value(argv, i, tok));
    } else if (tok == "--lam") {
      a.config.lam = std::stod(require_value(argv, i, tok));
    } else if (tok == "--beta") {
      a.config.beta = std::stod(require_value(argv, i, tok));
    } else if (tok == "--seed") {
      a.config.seed = std::stoi(require_value(argv, i, tok));
    } else if (tok == "--no-bias") {
      a.config.add_bias_feature = false;
    } else if (tok == "--steps") {
      a.steps = std::stoi(require_value(argv, i, tok));
    } else if (tok == "--manual-patch") {
      a.manual_patch = std::stoi(require_value(argv, i, tok));
    } else if (tok == "--steps-before-render") {
      a.steps_before_render = std::stoi(require_value(argv, i, tok));
    } else if (tok == "--out-dir") {
      a.out_dir = require_value(argv, i, tok);
    } else if (tok == "--player-mesh") {
      a.player_mesh = require_value(argv, i, tok);
    } else {
      throw std::runtime_error("Unknown option: " + tok);
    }
  }

  if (cmd == "simulate" && a.steps < 0) {
    throw std::runtime_error("--steps must be >= 0");
  }
  if (cmd == "render" && a.steps_before_render < 0) {
    throw std::runtime_error("--steps-before-render must be >= 0");
  }

  return a;
}

int cmd_simulate(const Args& a) {
  auto sim = sb::SpectralBanditSimulator::from_config(a.config);

  std::cout << "Mesh vertices: " << sim.mesh().n_vertices() << ", faces: " << sim.mesh().n_faces() << "\n";
  std::cout << "Patches: " << sim.n_patches() << ", spectral dim: " << sim.patch_contexts().cols() << "\n";
  const int best = sim.oracle_best_patch();
  std::cout << "Oracle best patch: " << best << ", p=" << std::fixed << std::setprecision(3)
            << sim.hidden_patch_probs()(best) << "\n";

  for (int i = 0; i < a.steps; ++i) {
    sb::StepResult r = sim.step(a.manual_patch);
    std::cout << "t=" << std::setw(3) << std::setfill('0') << r.t << " patch=" << std::setw(3) << r.chosen_patch
              << " reward=" << r.reward << std::setfill(' ') << std::fixed << std::setprecision(4)
              << " cost=" << r.travel_cost << " mu=" << r.predicted_mu << " sigma=" << r.predicted_sigma
              << " score=" << r.decision_score << " oracle_p=" << r.oracle_prob << "\n";
  }

  Eigen::VectorXd rewards(sim.history().size());
  for (Eigen::Index i = 0; i < rewards.size(); ++i) {
    rewards(i) = sim.history()[static_cast<std::size_t>(i)].reward;
  }

  const double mean_reward = (rewards.size() > 0) ? rewards.mean() : 0.0;

  std::cout << "---\n";
  std::cout << "Cumulative reward: " << sim.cumulative_reward() << "\n";
  std::cout << "Mean reward: " << std::fixed << std::setprecision(4) << mean_reward << "\n";
  return 0;
}

int cmd_render(const Args& a) {
#ifdef SB_WITH_VIEWER
  std::cout << "Initializing simulator and renderer... this can take time depending on mesh/patches." << std::endl;
  return sb::render_heatmaps_png(a.config, a.steps_before_render, a.out_dir, a.player_mesh);
#else
  (void)a;
  throw std::runtime_error("Viewer build disabled: reconfigure with SB_BUILD_VIEWER=ON");
#endif
}

int cmd_viewer(const Args& a) {
#ifdef SB_WITH_VIEWER
  std::cout << "Loading viewer... precomputing geodesic patches and spectral basis (please wait)." << std::endl;
  sb::run_game(a.config, a.player_mesh, true);
  return 0;
#else
  (void)a;
  throw std::runtime_error("Viewer build disabled: reconfigure with SB_BUILD_VIEWER=ON");
#endif
}

int cmd_game(const Args& a) {
#ifdef SB_WITH_VIEWER
  std::cout << "Loading game... precomputing geodesic patches and spectral basis (please wait)." << std::endl;
  sb::run_game(a.config, a.player_mesh, true);
  return 0;
#else
  (void)a;
  throw std::runtime_error("Viewer build disabled: reconfigure with SB_BUILD_VIEWER=ON");
#endif
}

int cmd_game3d(const Args& a) {
#ifdef SB_WITH_VIEWER
  std::cout << "Loading game3d... precomputing geodesic patches and spectral basis (please wait)." << std::endl;
  sb::run_game(a.config, a.player_mesh, true);
  return 0;
#else
  (void)a;
  throw std::runtime_error("Viewer build disabled: reconfigure with SB_BUILD_VIEWER=ON");
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 1;
    }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help") {
      usage();
      return 0;
    }

    const std::vector<std::string> args(argv + 2, argv + argc);
    const Args parsed = parse_args(cmd, args);

    if (cmd == "simulate") {
      return cmd_simulate(parsed);
    }
    if (cmd == "render") {
      return cmd_render(parsed);
    }
    if (cmd == "viewer") {
      return cmd_viewer(parsed);
    }
    if (cmd == "game") {
      return cmd_game(parsed);
    }
    if (cmd == "game3d") {
      return cmd_game3d(parsed);
    }

    throw std::runtime_error("Unknown command: " + cmd);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}

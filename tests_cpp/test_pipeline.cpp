#include "spectral_bandit/geometry.hpp"
#include "spectral_bandit/heat.hpp"
#include "spectral_bandit/mesh.hpp"
#include "spectral_bandit/patchification.hpp"
#include "spectral_bandit/pathfinding.hpp"
#include "spectral_bandit/simulator.hpp"
#include "spectral_bandit/spectral.hpp"
#include "spectral_bandit/viewer.hpp"

#include <Eigen/Core>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sb = spectral_bandit;

namespace {

void expect(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error("Test failed: " + msg);
  }
}

std::filesystem::path source_root() {
#ifdef SB_SOURCE_DIR
  return std::filesystem::path(SB_SOURCE_DIR);
#else
  return std::filesystem::current_path();
#endif
}

void test_obj_load_and_operators() {
  const auto player_obj = source_root() / "player" / "player_texture.obj";

  auto mesh = sb::load_obj(player_obj);
  expect(mesh.n_vertices() > 100, "mesh vertices > 100");
  expect(mesh.n_faces() > 100, "mesh faces > 100");

  auto [mass_diag, mass_mat] = sb::mass_matrix(mesh);
  auto lap = sb::cotangent_laplacian(mesh);

  expect((mass_diag.array() > 0.0).all(), "all masses positive");
  expect(lap.rows() == mesh.n_vertices() && lap.cols() == mesh.n_vertices(), "laplacian shape");

  sb::SparseMatrix sym = lap - sb::SparseMatrix(lap.transpose());
  double sq = 0.0;
  for (int k = 0; k < sym.outerSize(); ++k) {
    for (sb::SparseMatrix::InnerIterator it(sym, k); it; ++it) {
      sq += it.value() * it.value();
    }
  }
  expect(sq < 1e-6, "laplacian symmetry error < 1e-6");

  (void)mass_mat;
}

void test_heat_method_and_patchification() {
  const auto player_obj = source_root() / "player" / "player_texture.obj";

  auto mesh = sb::load_obj(player_obj);
  auto [mass_diag, _] = sb::mass_matrix(mesh);
  auto lap = sb::cotangent_laplacian(mesh);

  sb::HeatMethodGeodesics heat(mesh, lap, mass_diag);
  Eigen::VectorXd d = heat.distance_from(0);

  expect(d.size() == mesh.n_vertices(), "distance size");
  expect(d.allFinite(), "distance finite");

  Eigen::Index argmin = 0;
  d.minCoeff(&argmin);
  expect(static_cast<int>(argmin) == 0, "distance source at argmin");

  auto patch = sb::geodesic_patchify(mesh, heat, mass_diag, 6, 0);
  expect(patch.centers.size() == 6, "6 centers");
  expect(patch.vertex_labels.size() == mesh.n_vertices(), "vertex labels size");
  expect(patch.face_labels.size() == mesh.n_faces(), "face labels size");

  for (const auto& p : patch.patches) {
    expect(p.size() > 0, "non-empty patch");
  }
}

void test_spectral_patch_embeddings_and_dijkstra() {
  const auto player_obj = source_root() / "player" / "player_texture.obj";

  auto mesh = sb::load_obj(player_obj);
  auto [mass_diag, mass_mat] = sb::mass_matrix(mesh);
  auto lap = sb::cotangent_laplacian(mesh);

  sb::HeatMethodGeodesics heat(mesh, lap, mass_diag);
  auto patch = sb::geodesic_patchify(mesh, heat, mass_diag, 5, 1);

  auto spec = sb::spectral_embedding(lap, mass_mat, 5, true);
  auto patch_x = sb::patch_embeddings(spec.vertex_embeddings, patch.vertex_labels, mass_diag, 5);

  expect(patch_x.rows() == 5, "patch embeddings rows");
  expect(patch_x.cols() == spec.vertex_embeddings.cols(), "patch embeddings cols");
  expect(patch_x.allFinite(), "patch embeddings finite");

  auto adj = sb::build_graph_adjacency(mesh);
  auto [path, cost] = sb::dijkstra_shortest_path(adj, patch.center_vertices(0), patch.center_vertices(1));

  expect(path.size() >= 2, "dijkstra path size >=2");
  expect(cost >= 0.0, "dijkstra cost >=0");
}

void test_full_simulator_short_run() {
  const auto player_obj = source_root() / "player" / "player_texture.obj";

  sb::SimulatorConfig cfg;
  cfg.mesh_path = player_obj.string();
  cfg.n_patches = 6;
  cfg.spectral_dim = 5;
  cfg.alpha = 1.2;
  cfg.lam = 1.0;
  cfg.beta = 0.0;
  cfg.seed = 7;

  auto sim = sb::SpectralBanditSimulator::from_config(cfg);
  auto out = sim.run(4);

  expect(static_cast<int>(out.size()) == 4, "4 steps output");
  expect(sim.t() == 4, "t=4");
  expect(sim.cumulative_reward() >= 0 && sim.cumulative_reward() <= 4, "cumulative reward bounds");

  auto heatmaps = sim.heatmap_values();
  expect(heatmaps.at("ucb").size() == sim.n_patches(), "ucb size per patch");
  expect(heatmaps.at("ucb").allFinite(), "ucb finite");
}

void test_game_render_assets_parse() {
  const auto planet_obj = source_root() / "planet" / "planet_texture.obj";
  auto info = sb::load_render_mesh_info(planet_obj.string());

  expect(info.n_vertices > 1000, "planet render n_vertices > 1000");
  expect(info.face_count > 1000, "planet render face_count > 1000");

  Eigen::VectorXd v(3);
  v << 0.0, 0.5, 1.0;
  Eigen::MatrixXd c = sb::visibility_shadow_colors(v);

  expect(c.rows() == 3 && c.cols() == 3, "visibility colors shape 3x3");
  expect((c.col(0) - c.col(1)).cwiseAbs().maxCoeff() < 1e-12, "r==g");
  expect((c.col(1) - c.col(2)).cwiseAbs().maxCoeff() < 1e-12, "g==b");
  expect(c(0, 0) < c(1, 0) && c(1, 0) < c(2, 0), "shadow -> illuminated monotonic");
}

}  // namespace

int main() {
  try {
    test_obj_load_and_operators();
    test_heat_method_and_patchification();
    test_spectral_patch_embeddings_and_dijkstra();
    test_full_simulator_short_run();
    test_game_render_assets_parse();

    std::cout << "All tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}

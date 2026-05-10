#include "spectral_bandit/viewer.hpp"

#include "spectral_bandit/geometry.hpp"

#include <raylib.h>
#include <raymath.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace spectral_bandit {
namespace {

constexpr float kPi = 3.14159265358979323846F;

struct ObjRef {
  int v = -1;
  int vt = -1;
  int vn = -1;
};

struct ObjRenderData {
  std::vector<Vector3> vertices;
  std::vector<Vector2> texcoords;
  std::vector<Vector3> normals;
  std::vector<std::array<ObjRef, 3>> triangles;
};

struct UiButton {
  std::string id;
  std::string label;
  Rectangle rect;
};

struct BuiltModel {
  Model model{};
  int face_count = 0;
};

std::string trim(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  std::size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

ObjRef parse_obj_ref(const std::string& tok) {
  ObjRef out;

  const auto p1 = tok.find('/');
  if (p1 == std::string::npos) {
    out.v = std::stoi(tok) - 1;
    return out;
  }

  out.v = std::stoi(tok.substr(0, p1)) - 1;
  const auto p2 = tok.find('/', p1 + 1);

  if (p2 == std::string::npos) {
    const std::string vt = tok.substr(p1 + 1);
    if (!vt.empty()) {
      out.vt = std::stoi(vt) - 1;
    }
    return out;
  }

  const std::string vt = tok.substr(p1 + 1, p2 - p1 - 1);
  const std::string vn = tok.substr(p2 + 1);
  if (!vt.empty()) {
    out.vt = std::stoi(vt) - 1;
  }
  if (!vn.empty()) {
    out.vn = std::stoi(vn) - 1;
  }

  return out;
}

ObjRenderData parse_obj_render(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open OBJ render file: " + path.string());
  }

  ObjRenderData data;

  std::string raw;
  while (std::getline(in, raw)) {
    const std::string line = trim(raw);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    if (line.rfind("v ", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      iss >> tag >> x >> y >> z;
      if (!iss.fail()) {
        data.vertices.push_back(Vector3{x, y, z});
      }
      continue;
    }

    if (line.rfind("vt ", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      float u = 0.0F;
      float v = 0.0F;
      iss >> tag >> u >> v;
      if (!iss.fail()) {
        data.texcoords.push_back(Vector2{u, v});
      }
      continue;
    }

    if (line.rfind("vn ", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      iss >> tag >> x >> y >> z;
      if (!iss.fail()) {
        data.normals.push_back(Vector3{x, y, z});
      }
      continue;
    }

    if (line.rfind("f ", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      iss >> tag;

      std::vector<ObjRef> poly;
      std::string tok;
      while (iss >> tok) {
        poly.push_back(parse_obj_ref(tok));
      }

      if (poly.size() < 3) {
        continue;
      }

      for (std::size_t i = 1; i + 1 < poly.size(); ++i) {
        data.triangles.push_back({poly[0], poly[i], poly[i + 1]});
      }
    }
  }

  if (data.vertices.empty() || data.triangles.empty()) {
    throw std::runtime_error("Invalid OBJ render data: " + path.string());
  }

  return data;
}

std::filesystem::path resolve_texture_path(const std::filesystem::path& obj_path) {
  auto stem_png = obj_path;
  stem_png.replace_extension(".png");
  if (std::filesystem::exists(stem_png)) {
    return stem_png;
  }

  for (const auto& entry : std::filesystem::directory_iterator(obj_path.parent_path())) {
    if (entry.is_regular_file() && entry.path().extension() == ".png") {
      return entry.path();
    }
  }

  throw std::runtime_error("No texture PNG found near " + obj_path.string());
}

Vector2 spherical_uv(Vector3 p) {
  const Vector3 pn = Vector3Normalize(p);
  const float u = 0.5F + std::atan2(pn.z, pn.x) / (2.0F * kPi);
  const float v = 0.5F - std::asin(std::clamp(pn.y, -1.0F, 1.0F)) / kPi;
  return Vector2{u, v};
}

BuiltModel build_model_from_obj(const std::filesystem::path& obj_path) {
  const ObjRenderData obj = parse_obj_render(obj_path);
  const int face_count = static_cast<int>(obj.triangles.size());
  const int vertex_count = face_count * 3;

  ::Mesh mesh{};
  mesh.vertexCount = vertex_count;
  mesh.triangleCount = face_count;

  mesh.vertices = static_cast<float*>(MemAlloc(static_cast<std::size_t>(vertex_count) * 3U * sizeof(float)));
  mesh.normals = static_cast<float*>(MemAlloc(static_cast<std::size_t>(vertex_count) * 3U * sizeof(float)));
  mesh.texcoords = static_cast<float*>(MemAlloc(static_cast<std::size_t>(vertex_count) * 2U * sizeof(float)));
  mesh.colors = static_cast<unsigned char*>(MemAlloc(static_cast<std::size_t>(vertex_count) * 4U * sizeof(unsigned char)));

  if (mesh.vertices == nullptr || mesh.normals == nullptr || mesh.texcoords == nullptr || mesh.colors == nullptr) {
    throw std::runtime_error("Mesh allocation failed");
  }

  int out_vi = 0;
  for (const auto& tri : obj.triangles) {
    const Vector3 p0 = obj.vertices[static_cast<std::size_t>(tri[0].v)];
    const Vector3 p1 = obj.vertices[static_cast<std::size_t>(tri[1].v)];
    const Vector3 p2 = obj.vertices[static_cast<std::size_t>(tri[2].v)];
    const Vector3 fn = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(p1, p0), Vector3Subtract(p2, p0)));

    for (int k = 0; k < 3; ++k) {
      const ObjRef ref = tri[static_cast<std::size_t>(k)];
      const Vector3 p = obj.vertices[static_cast<std::size_t>(ref.v)];

      Vector3 n = fn;
      if (ref.vn >= 0 && ref.vn < static_cast<int>(obj.normals.size())) {
        n = Vector3Normalize(obj.normals[static_cast<std::size_t>(ref.vn)]);
      }

      Vector2 uv = spherical_uv(p);
      if (ref.vt >= 0 && ref.vt < static_cast<int>(obj.texcoords.size())) {
        uv = obj.texcoords[static_cast<std::size_t>(ref.vt)];
      }

      mesh.vertices[out_vi * 3 + 0] = p.x;
      mesh.vertices[out_vi * 3 + 1] = p.y;
      mesh.vertices[out_vi * 3 + 2] = p.z;

      mesh.normals[out_vi * 3 + 0] = n.x;
      mesh.normals[out_vi * 3 + 1] = n.y;
      mesh.normals[out_vi * 3 + 2] = n.z;

      mesh.texcoords[out_vi * 2 + 0] = uv.x;
      mesh.texcoords[out_vi * 2 + 1] = uv.y;

      mesh.colors[out_vi * 4 + 0] = 255;
      mesh.colors[out_vi * 4 + 1] = 255;
      mesh.colors[out_vi * 4 + 2] = 255;
      mesh.colors[out_vi * 4 + 3] = 255;

      out_vi += 1;
    }
  }

  UploadMesh(&mesh, true);
  Model model = LoadModelFromMesh(mesh);

  BuiltModel out;
  out.model = model;
  out.face_count = face_count;
  return out;
}

int material_map_albedo() {
#ifdef MATERIAL_MAP_ALBEDO
  return MATERIAL_MAP_ALBEDO;
#else
  return MATERIAL_MAP_DIFFUSE;
#endif
}

HeatmapMode next_mode_from_id(const std::string& id, HeatmapMode current) {
  if (id == "mode_texture") {
    return HeatmapMode::Texture;
  }
  if (id == "mode_exploration") {
    return HeatmapMode::Exploration;
  }
  if (id == "mode_exploitation") {
    return HeatmapMode::Exploitation;
  }
  if (id == "mode_ucb") {
    return HeatmapMode::UCB;
  }
  if (id == "mode_oracle") {
    return HeatmapMode::Oracle;
  }
  if (id == "mode_seen") {
    return HeatmapMode::Seen;
  }
  return current;
}

const char* mode_label(HeatmapMode mode) {
  switch (mode) {
    case HeatmapMode::Texture:
      return "Texture";
    case HeatmapMode::Exploration:
      return "Exploration";
    case HeatmapMode::Exploitation:
      return "Exploitation";
    case HeatmapMode::UCB:
      return "UCB";
    case HeatmapMode::Oracle:
      return "Oracle";
    case HeatmapMode::Seen:
      return "Seen";
  }
  return "UCB";
}

constexpr const char* kVisualVs = R"GLSL(
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

out vec2 fragTexCoord;
out vec3 fragPos;
out vec3 fragNormal;
out vec4 fragColor;

uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;

void main() {
  fragTexCoord = vertexTexCoord;
  fragColor = vertexColor;
  fragPos = vec3(matModel * vec4(vertexPosition, 1.0));
  fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 0.0)));
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)GLSL";

constexpr const char* kVisualFs = R"GLSL(
#version 330
in vec2 fragTexCoord;
in vec3 fragPos;
in vec3 fragNormal;
in vec4 fragColor;
out vec4 finalColor;

uniform sampler2D texture0;
uniform vec4 colDiffuse;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 fillLightDir;
uniform vec3 fillLightColor;
uniform vec3 viewPos;
uniform vec3 ambientColor;
uniform float specPower;
uniform float specStrength;
uniform float rimStrength;

void main() {
  vec4 texel = texture(texture0, fragTexCoord) * colDiffuse * fragColor;
  vec3 albedo = texel.rgb;
  float visMask = clamp((fragColor.r + fragColor.g + fragColor.b) / 3.0, 0.0, 1.0);
  float shinyMask = smoothstep(0.25, 1.0, visMask);

  vec3 N = normalize(fragNormal);
  vec3 V = normalize(viewPos - fragPos);

  // Key point light.
  vec3 L = normalize(lightPos - fragPos);
  float dist = length(lightPos - fragPos);
  float atten = 1.0 / (1.0 + 0.18 * dist + 0.035 * dist * dist);
  float diff = max(dot(N, L), 0.0) * atten;

  // Soft fill directional light to avoid overly dark sides.
  vec3 Lf = normalize(-fillLightDir);
  float fill = max(dot(N, Lf), 0.0);

  vec3 H = normalize(L + V);
  vec3 Hf = normalize(Lf + V);
  float spec = pow(max(dot(N, H), 0.0), specPower) * specStrength * atten;
  float specFill = pow(max(dot(N, Hf), 0.0), specPower * 0.55) * (specStrength * 0.28);
  float rim = pow(1.0 - max(dot(N, V), 0.0), 2.4) * rimStrength;

  vec3 lit = ambientColor * albedo;
  lit += diff * lightColor * albedo;
  lit += fill * fillLightColor * albedo;
  lit += (spec + rim) * lightColor * shinyMask;
  lit += specFill * fillLightColor * shinyMask;

  // Gentle filmic/gamma shaping for crisper highlights without burning colors.
  lit = lit / (lit + vec3(1.0));
  lit = pow(lit, vec3(1.0 / 2.2));

  finalColor = vec4(lit, texel.a);
}
)GLSL";

class GameApp {
 public:
  GameApp(const SimulatorConfig& config, const std::string& player_mesh_path, bool show_panel)
      : sim_(SpectralBanditSimulator::from_config(config)),
        mesh_normals_(vertex_normals(sim_.mesh())),
        show_panel_(show_panel),
        panel_width_(360),
        ui_margin_(12) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(1500, 900, "Spectral Bandit C++");
    SetTargetFPS(240);

    const std::filesystem::path planet_obj(config.mesh_path);
    const BuiltModel planet = build_model_from_obj(planet_obj);
    if (planet.face_count != sim_.mesh().n_faces()) {
      throw std::runtime_error("Face mismatch between render OBJ and simulator mesh");
    }
    planet_model_ = planet.model;

    const auto planet_tex_path = resolve_texture_path(planet_obj);
    planet_texture_ = LoadTexture(planet_tex_path.string().c_str());
    SetMaterialTexture(&planet_model_.materials[0], material_map_albedo(), planet_texture_);
    SetTextureFilter(planet_texture_, TEXTURE_FILTER_ANISOTROPIC_16X);

    has_player_ = false;
    try {
      player_model_ = LoadModel(player_mesh_path.c_str());
      const auto player_tex_path = resolve_texture_path(std::filesystem::path(player_mesh_path));
      player_texture_ = LoadTexture(player_tex_path.string().c_str());
      SetMaterialTexture(&player_model_.materials[0], material_map_albedo(), player_texture_);
      SetTextureFilter(player_texture_, TEXTURE_FILTER_ANISOTROPIC_16X);
      has_player_ = true;
    } catch (...) {
      has_player_ = false;
    }

    avg_edge_len_ = std::max(average_edge_length(sim_.mesh()), 1e-4);
    move_speed_ = std::max(0.6, 3.5 * avg_edge_len_);
    path_radius_ = std::clamp(0.26F * static_cast<float>(avg_edge_len_), 0.0032F, 0.011F);
    player_clearance_ = std::clamp(0.12F * static_cast<float>(avg_edge_len_), 0.0015F, 0.0065F);
    player_scale_ = 0.12F;
    player_base_height_ = estimate_player_base_height();

    const int cv = sim_.current_vertex();
    const Eigen::Vector3d p = sim_.mesh().vertices.row(cv);
    player_position_ = Vector3{static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
    player_normal_ = normal_at_vertex(cv);
    player_forward_ = Vector3{1.0F, 0.0F, 0.0F};

    cam_yaw_ = 0.7F;
    cam_pitch_ = 0.25F;
    cam_dist_ = 3.2F;

    dragging_ = false;
    ui_drag_blocked_ = false;

    mode_ = HeatmapMode::Texture;
    selected_patch_ = 0;
    auto_mode_ = false;
    moving_ = false;
    move_progress_idx_ = 0;
    move_progress_dist_ = 0.0;

    init_visual_shader();
    rebuild_camera();
    build_ui_buttons();
    update_planet_colors();
  }

  ~GameApp() {
    if (has_visual_shader_) {
      UnloadShader(visual_shader_);
    }
    if (has_player_) {
      UnloadModel(player_model_);
      UnloadTexture(player_texture_);
    }
    UnloadModel(planet_model_);
    UnloadTexture(planet_texture_);
    CloseWindow();
  }

  void run() {
    while (!WindowShouldClose()) {
      const float dt = GetFrameTime();
      handle_input();
      update(dt);
      draw();
    }
  }

  void run_steps(int steps) {
    for (int i = 0; i < steps; ++i) {
      const int patch = recommended_patch(true);
      queue_step(patch);
      // Snap movement to destination for offline rendering.
      finish_current_movement();
    }
  }

  void set_mode(HeatmapMode mode) {
    mode_ = mode;
    update_planet_colors();
  }

  void draw_once() { draw(); }

 private:
  void init_visual_shader() {
    has_visual_shader_ = false;
    visual_shader_ = LoadShaderFromMemory(kVisualVs, kVisualFs);
    if (visual_shader_.id <= 0) {
      return;
    }

    light_pos_ = Vector3{2.65F, 1.62F, 2.35F};
    fill_light_dir_ = Vector3Normalize(Vector3{-1.0F, -0.35F, -0.55F});
    light_color_ = Vector3{1.20F, 1.10F, 1.00F};
    fill_light_color_ = Vector3{0.28F, 0.35F, 0.50F};
    ambient_color_ = Vector3{0.10F, 0.13F, 0.18F};
    spec_power_ = 76.0F;
    spec_strength_ = 0.62F;
    rim_strength_ = 0.36F;

    loc_light_pos_ = GetShaderLocation(visual_shader_, "lightPos");
    loc_light_color_ = GetShaderLocation(visual_shader_, "lightColor");
    loc_fill_light_dir_ = GetShaderLocation(visual_shader_, "fillLightDir");
    loc_fill_light_color_ = GetShaderLocation(visual_shader_, "fillLightColor");
    loc_view_pos_ = GetShaderLocation(visual_shader_, "viewPos");
    loc_ambient_color_ = GetShaderLocation(visual_shader_, "ambientColor");
    loc_spec_power_ = GetShaderLocation(visual_shader_, "specPower");
    loc_spec_strength_ = GetShaderLocation(visual_shader_, "specStrength");
    loc_rim_strength_ = GetShaderLocation(visual_shader_, "rimStrength");

    if (loc_light_pos_ >= 0) {
      SetShaderValue(visual_shader_, loc_light_pos_, &light_pos_, SHADER_UNIFORM_VEC3);
    }
    if (loc_light_color_ >= 0) {
      SetShaderValue(visual_shader_, loc_light_color_, &light_color_, SHADER_UNIFORM_VEC3);
    }
    if (loc_fill_light_dir_ >= 0) {
      SetShaderValue(visual_shader_, loc_fill_light_dir_, &fill_light_dir_, SHADER_UNIFORM_VEC3);
    }
    if (loc_fill_light_color_ >= 0) {
      SetShaderValue(visual_shader_, loc_fill_light_color_, &fill_light_color_, SHADER_UNIFORM_VEC3);
    }
    if (loc_ambient_color_ >= 0) {
      SetShaderValue(visual_shader_, loc_ambient_color_, &ambient_color_, SHADER_UNIFORM_VEC3);
    }
    if (loc_spec_power_ >= 0) {
      SetShaderValue(visual_shader_, loc_spec_power_, &spec_power_, SHADER_UNIFORM_FLOAT);
    }
    if (loc_spec_strength_ >= 0) {
      SetShaderValue(visual_shader_, loc_spec_strength_, &spec_strength_, SHADER_UNIFORM_FLOAT);
    }
    if (loc_rim_strength_ >= 0) {
      SetShaderValue(visual_shader_, loc_rim_strength_, &rim_strength_, SHADER_UNIFORM_FLOAT);
    }

    for (int m = 0; m < planet_model_.materialCount; ++m) {
      planet_model_.materials[m].shader = visual_shader_;
    }
    if (has_player_) {
      for (int m = 0; m < player_model_.materialCount; ++m) {
        player_model_.materials[m].shader = visual_shader_;
      }
    }

    has_visual_shader_ = true;
  }

  void update_visual_uniforms() {
    if (!has_visual_shader_) {
      return;
    }

    const float t = static_cast<float>(GetTime());
    light_pos_ = Vector3{2.65F * std::cos(0.23F * t), 1.62F + 0.24F * std::sin(0.43F * t),
                         2.65F * std::sin(0.23F * t)};

    if (loc_light_pos_ >= 0) {
      SetShaderValue(visual_shader_, loc_light_pos_, &light_pos_, SHADER_UNIFORM_VEC3);
    }
    if (loc_view_pos_ >= 0) {
      SetShaderValue(visual_shader_, loc_view_pos_, &camera_.position, SHADER_UNIFORM_VEC3);
    }
  }

  void rebuild_camera() {
    const float cp = std::cos(cam_pitch_);
    const float sp = std::sin(cam_pitch_);
    const float cy = std::cos(cam_yaw_);
    const float sy = std::sin(cam_yaw_);

    camera_.position = Vector3{cam_dist_ * cp * cy, cam_dist_ * sp, cam_dist_ * cp * sy};
    camera_.target = Vector3{0.0F, 0.0F, 0.0F};
    camera_.up = Vector3{0.0F, 1.0F, 0.0F};
    camera_.fovy = 50.0F;
    camera_.projection = CAMERA_PERSPECTIVE;
  }

  float estimate_player_base_height() const {
    if (!has_player_) {
      return 0.0F;
    }

    float y_min = std::numeric_limits<float>::infinity();
    for (int mi = 0; mi < player_model_.meshCount; ++mi) {
      const ::Mesh& m = player_model_.meshes[mi];
      for (int vi = 0; vi < m.vertexCount; ++vi) {
        const float y = m.vertices[vi * 3 + 1];
        y_min = std::min(y_min, y);
      }
    }

    if (!std::isfinite(y_min)) {
      return 0.0F;
    }
    return -y_min;
  }

  Vector3 normal_at_vertex(int vertex_idx) const {
    const Eigen::Vector3d n = mesh_normals_.row(vertex_idx);
    return Vector3Normalize(Vector3{static_cast<float>(n.x()), static_cast<float>(n.y()), static_cast<float>(n.z())});
  }

  Eigen::VectorXd seen_values() const {
    Eigen::VectorXd seen = Eigen::VectorXd::Zero(sim_.n_patches());
    for (const auto& s : sim_.history()) {
      seen(s.chosen_patch) += 1.0;
    }
    const double mx = seen.maxCoeff();
    if (mx > 0.0) {
      seen /= mx;
    }
    return seen;
  }

  Eigen::VectorXd mode_values() const {
    const auto vals = sim_.heatmap_values();

    switch (mode_) {
      case HeatmapMode::Texture:
        return Eigen::VectorXd::Ones(sim_.n_patches());
      case HeatmapMode::Exploration:
        return vals.at("explore");
      case HeatmapMode::Exploitation:
        return vals.at("exploit");
      case HeatmapMode::UCB:
        return vals.at("ucb");
      case HeatmapMode::Oracle:
        return vals.at("oracle");
      case HeatmapMode::Seen:
        return seen_values();
    }
    return vals.at("ucb");
  }

  void update_planet_colors() {
    const int face_count = sim_.mesh().n_faces();
    unsigned char* colors = planet_model_.meshes[0].colors;

    if (colors == nullptr) {
      return;
    }

    Eigen::VectorXd vis(face_count);
    if (mode_ == HeatmapMode::Texture) {
      vis.setOnes();
    } else {
      const Eigen::VectorXd patch_vals = mode_values();
      Eigen::VectorXd face_vals(face_count);
      for (int fi = 0; fi < face_count; ++fi) {
        face_vals(fi) = patch_vals(sim_.patchification().face_labels(fi));
      }

      const double vmin = face_vals.minCoeff();
      const double vmax = face_vals.maxCoeff();
      if (vmax - vmin < 1e-12) {
        vis.setConstant(0.01);
      } else {
        vis = (face_vals.array() - vmin) / (vmax - vmin);
        vis = 0.01 + 0.99 * vis.array().pow(2.2);
      }
    }

    for (int fi = 0; fi < face_count; ++fi) {
      double f = vis(fi);
      if (selected_patch_ >= 0 && selected_patch_ < sim_.n_patches()) {
        if (sim_.patchification().face_labels(fi) == selected_patch_) {
          f = std::min(1.0, f * 1.35);
        }
      }

      const auto c = static_cast<unsigned char>(std::round(255.0 * std::clamp(f, 0.0, 1.0)));
      for (int k = 0; k < 3; ++k) {
        const int vi = fi * 3 + k;
        colors[vi * 4 + 0] = c;
        colors[vi * 4 + 1] = c;
        colors[vi * 4 + 2] = c;
        colors[vi * 4 + 3] = 255;
      }
    }

    UpdateMeshBuffer(planet_model_.meshes[0], 3, colors,
                     planet_model_.meshes[0].vertexCount * 4 * static_cast<int>(sizeof(unsigned char)), 0);
  }

  int recommended_patch(bool prefer_move) const {
    const auto snap = sim_.current_scores();
    std::vector<int> order(static_cast<std::size_t>(snap.decision.size()));
    for (Eigen::Index i = 0; i < snap.decision.size(); ++i) {
      order[static_cast<std::size_t>(i)] = static_cast<int>(i);
    }

    std::sort(order.begin(), order.end(), [&snap](int a, int b) { return snap.decision(a) > snap.decision(b); });

    if (prefer_move) {
      const int cur_v = sim_.current_vertex();
      for (int idx : order) {
        if (sim_.patchification().center_vertices(idx) != cur_v) {
          return idx;
        }
      }
    }

    return order.empty() ? 0 : order[0];
  }

  void queue_step(int patch) {
    if (moving_) {
      return;
    }

    StepResult res = sim_.step(patch);
    move_path_.clear();
    move_path_.reserve(static_cast<std::size_t>(res.path.size()));
    for (Eigen::Index i = 0; i < res.path.size(); ++i) {
      move_path_.push_back(res.path(i));
    }

    move_progress_idx_ = 0;
    move_progress_dist_ = 0.0;
    moving_ = move_path_.size() >= 2;

    if (!moving_) {
      const Eigen::Vector3d p = sim_.mesh().vertices.row(sim_.current_vertex());
      player_position_ = Vector3{static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
      player_normal_ = normal_at_vertex(sim_.current_vertex());
    }

    last_action_info_ = "t=" + std::to_string(res.t) + " patch=" + std::to_string(res.chosen_patch) +
                        " reward=" + std::to_string(res.reward) + " oracle_p=" + std::to_string(res.oracle_prob) +
                        " cost=" + std::to_string(res.travel_cost);

    update_planet_colors();
  }

  void finish_current_movement() {
    if (move_path_.empty()) {
      return;
    }
    const int last = move_path_.back();
    const Eigen::Vector3d p = sim_.mesh().vertices.row(last);
    player_position_ = Vector3{static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())};
    player_normal_ = normal_at_vertex(last);
    moving_ = false;
    move_progress_idx_ = 0;
    move_progress_dist_ = 0.0;
  }

  void update_movement(float dt) {
    if (!moving_ || move_path_.size() < 2) {
      return;
    }

    float remaining = dt;

    while (remaining > 1e-7F && move_progress_idx_ + 1 < static_cast<int>(move_path_.size())) {
      const int a_idx = move_path_[static_cast<std::size_t>(move_progress_idx_)];
      const int b_idx = move_path_[static_cast<std::size_t>(move_progress_idx_ + 1)];

      const Eigen::Vector3d a_e = sim_.mesh().vertices.row(a_idx);
      const Eigen::Vector3d b_e = sim_.mesh().vertices.row(b_idx);
      const Vector3 a{static_cast<float>(a_e.x()), static_cast<float>(a_e.y()), static_cast<float>(a_e.z())};
      const Vector3 b{static_cast<float>(b_e.x()), static_cast<float>(b_e.y()), static_cast<float>(b_e.z())};
      const Vector3 seg = Vector3Subtract(b, a);
      const float seg_len = Vector3Length(seg);

      if (seg_len < 1e-8F) {
        move_progress_idx_ += 1;
        move_progress_dist_ = 0.0;
        continue;
      }

      const float dist_left = seg_len - static_cast<float>(move_progress_dist_);
      const float step_dist = static_cast<float>(move_speed_) * remaining;

      if (step_dist >= dist_left) {
        player_position_ = b;
        player_normal_ = normal_at_vertex(b_idx);
        Vector3 tangent = Vector3Normalize(seg);
        tangent = Vector3Subtract(tangent, Vector3Scale(player_normal_, Vector3DotProduct(tangent, player_normal_)));
        if (Vector3Length(tangent) > 1e-8F) {
          player_forward_ = Vector3Normalize(tangent);
        }

        const float used_dt = dist_left / std::max(1e-8F, static_cast<float>(move_speed_));
        remaining -= used_dt;
        move_progress_idx_ += 1;
        move_progress_dist_ = 0.0;
      } else {
        move_progress_dist_ += step_dist;
        const float u = static_cast<float>(move_progress_dist_ / seg_len);
        player_position_ = Vector3Lerp(a, b, u);

        const Vector3 n_a = normal_at_vertex(a_idx);
        const Vector3 n_b = normal_at_vertex(b_idx);
        player_normal_ = Vector3Normalize(Vector3Lerp(n_a, n_b, u));

        Vector3 tangent = Vector3Normalize(seg);
        tangent = Vector3Subtract(tangent, Vector3Scale(player_normal_, Vector3DotProduct(tangent, player_normal_)));
        if (Vector3Length(tangent) > 1e-8F) {
          player_forward_ = Vector3Normalize(tangent);
        }

        remaining = 0.0F;
      }
    }

    if (move_progress_idx_ + 1 >= static_cast<int>(move_path_.size())) {
      moving_ = false;
    }
  }

  void build_ui_buttons() {
    buttons_.clear();
    const float x0 = static_cast<float>(ui_margin_ + 14);
    const float w = 156.0F;
    const float h = 34.0F;
    const float gap_x = 14.0F;
    const float gap_y = 8.0F;
    const float x1 = x0 + w + gap_x;

    float y = static_cast<float>(GetScreenHeight()) - 95.0F;

    const std::vector<std::tuple<std::string, std::string, int>> defs = {
        {"mode_texture", "Texture", 0},       {"mode_exploration", "Exploration", 1},
        {"mode_exploitation", "Exploitation", 0}, {"mode_ucb", "UCB", 1},
        {"mode_oracle", "Oracle", 0},         {"mode_seen", "Seen", 1},
        {"bandit_step", "Bandit Step", 0},    {"mine_selected", "Mine Selected", 1},
        {"select_reco", "Select Recommended", 0}, {"patch_prev", "Patch -", 1},
        {"patch_next", "Patch +", 0},         {"auto_x10", "Auto x10", 1},
        {"speed_down", "Speed -", 0},         {"speed_up", "Speed +", 1},
        {"toggle_auto", "Auto: OFF", 0},
    };

    for (const auto& [id, label, col] : defs) {
      const float x = (col == 0) ? x0 : x1;
      UiButton b;
      b.id = id;
      b.label = label;
      b.rect = Rectangle{x, y, w, h};
      buttons_.push_back(b);
      if (col == 1) {
        y -= h + gap_y;
      }
    }
  }

  void draw_ui() {
    if (!show_panel_) {
      return;
    }

    const int screen_h = GetScreenHeight();
    DrawRectangleRounded(Rectangle{static_cast<float>(ui_margin_), static_cast<float>(ui_margin_),
                                   static_cast<float>(panel_width_), static_cast<float>(screen_h - 2 * ui_margin_)},
                         0.04F, 10, Color{247, 247, 247, 245});
    DrawRectangleLinesEx(Rectangle{static_cast<float>(ui_margin_), static_cast<float>(ui_margin_),
                                   static_cast<float>(panel_width_), static_cast<float>(screen_h - 2 * ui_margin_)},
                         2.0F, Color{100, 100, 100, 255});

    DrawText("Galactic Prospector UI", ui_margin_ + 16, screen_h - ui_margin_ - 30, 22, BLACK);

    for (auto& b : buttons_) {
      Color fill = Color{225, 225, 225, 255};
      Color border = Color{100, 100, 100, 255};
      Color text = BLACK;

      const bool active_mode = ((b.id == "mode_texture" && mode_ == HeatmapMode::Texture) ||
                                (b.id == "mode_exploration" && mode_ == HeatmapMode::Exploration) ||
                                (b.id == "mode_exploitation" && mode_ == HeatmapMode::Exploitation) ||
                                (b.id == "mode_ucb" && mode_ == HeatmapMode::UCB) ||
                                (b.id == "mode_oracle" && mode_ == HeatmapMode::Oracle) ||
                                (b.id == "mode_seen" && mode_ == HeatmapMode::Seen));

      if (b.id == "toggle_auto") {
        b.label = auto_mode_ ? "Auto: ON" : "Auto: OFF";
      }

      if (active_mode || (b.id == "toggle_auto" && auto_mode_)) {
        fill = Color{40, 90, 170, 255};
        border = Color{20, 45, 95, 255};
        text = WHITE;
      }

      DrawRectangleRounded(b.rect, 0.2F, 8, fill);
      DrawRectangleLinesEx(b.rect, 2.0F, border);

      const int fs = 17;
      const int tw = MeasureText(b.label.c_str(), fs);
      DrawText(b.label.c_str(), static_cast<int>(b.rect.x + b.rect.width * 0.5F - static_cast<float>(tw) * 0.5F),
               static_cast<int>(b.rect.y + b.rect.height * 0.5F - static_cast<float>(fs) * 0.5F), fs, text);
    }

    const auto heat = sim_.heatmap_values();
    const Eigen::VectorXd seen = seen_values();
    const int p = std::clamp(selected_patch_, 0, sim_.n_patches() - 1);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Patch: " << p << "/" << (sim_.n_patches() - 1) << "\n";
    oss << "Oracle p: " << heat.at("oracle")(p) << "\n";
    oss << "UCB: " << heat.at("ucb")(p) << "\n";
    oss << "Mu: " << heat.at("exploit")(p) << "\n";
    oss << "Sigma: " << heat.at("explore")(p) << "\n";
    oss << "Seen score: " << seen(p) << "\n\n";
    oss << "Mode: " << mode_label(mode_) << "\n";
    oss << "Tour: " << sim_.t() << "\n";
    oss << "Gold: " << sim_.cumulative_reward() << "\n";
    oss << std::setprecision(2) << "Speed: " << move_speed_ << "\n";
    oss << std::setprecision(1) << "FPS: " << GetFPS() << "\n\n";
    oss << last_action_info_;

    const std::string info = oss.str();

    draw_wrapped_text_block(info, static_cast<float>(ui_margin_ + 16), 220.0F, static_cast<float>(panel_width_ - 32),
                            18.0F, 1.0F, static_cast<float>(screen_h - 260), BLACK);
  }

  void draw_wrapped_text_block(const std::string& text, float x, float y, float max_width, float font_size,
                               float spacing, float max_height, Color tint) const {
    const Font font = GetFontDefault();
    const float line_step = font_size + 5.0F;
    const int max_lines = std::max(1, static_cast<int>(std::floor(max_height / line_step)));

    std::vector<std::string> lines;
    lines.reserve(32);
    std::string current;
    current.reserve(256);

    for (char ch : text) {
      if (ch == '\n') {
        lines.push_back(current);
        current.clear();
        continue;
      }

      std::string candidate = current;
      candidate.push_back(ch);
      const float w = MeasureTextEx(font, candidate.c_str(), font_size, spacing).x;
      if (w > max_width && !current.empty()) {
        lines.push_back(current);
        current.clear();
        if (ch != ' ') {
          current.push_back(ch);
        }
      } else {
        current = std::move(candidate);
      }
    }
    if (!current.empty() || (!text.empty() && text.back() == '\n')) {
      lines.push_back(current);
    }

    const int draw_count = std::min<int>(static_cast<int>(lines.size()), max_lines);
    for (int i = 0; i < draw_count; ++i) {
      DrawTextEx(font, lines[static_cast<std::size_t>(i)].c_str(), Vector2{x, y + line_step * static_cast<float>(i)},
                 font_size, spacing, tint);
    }
  }

  bool is_in_panel(float x, float y) const {
    if (!show_panel_) {
      return false;
    }
    const Rectangle panel = Rectangle{static_cast<float>(ui_margin_), static_cast<float>(ui_margin_),
                                      static_cast<float>(panel_width_),
                                      static_cast<float>(GetScreenHeight() - 2 * ui_margin_)};
    return CheckCollisionPointRec(Vector2{x, y}, panel);
  }

  std::optional<std::string> button_at(float x, float y) const {
    if (!show_panel_) {
      return std::nullopt;
    }
    for (const auto& b : buttons_) {
      if (CheckCollisionPointRec(Vector2{x, y}, b.rect)) {
        return b.id;
      }
    }
    return std::nullopt;
  }

  void handle_action(const std::string& action) {
    if (action.rfind("mode_", 0) == 0) {
      mode_ = next_mode_from_id(action, mode_);
      update_planet_colors();
      return;
    }

    if (action == "bandit_step") {
      if (!moving_) {
        const int patch = recommended_patch(true);
        selected_patch_ = patch;
        queue_step(patch);
      }
      return;
    }

    if (action == "mine_selected") {
      if (!moving_) {
        int p = std::clamp(selected_patch_, 0, sim_.n_patches() - 1);
        if (sim_.patchification().center_vertices(p) == sim_.current_vertex()) {
          p = recommended_patch(true);
          selected_patch_ = p;
        }
        queue_step(p);
      }
      return;
    }

    if (action == "select_reco") {
      selected_patch_ = recommended_patch(false);
      update_planet_colors();
      return;
    }

    if (action == "patch_prev") {
      selected_patch_ = (selected_patch_ - 1 + sim_.n_patches()) % sim_.n_patches();
      update_planet_colors();
      return;
    }

    if (action == "patch_next") {
      selected_patch_ = (selected_patch_ + 1) % sim_.n_patches();
      update_planet_colors();
      return;
    }

    if (action == "auto_x10") {
      for (int i = 0; i < 10; ++i) {
        const int p = recommended_patch(true);
        queue_step(p);
        finish_current_movement();
      }
      selected_patch_ = recommended_patch(false);
      update_planet_colors();
      return;
    }

    if (action == "speed_down") {
      move_speed_ = std::max(0.2, move_speed_ - 0.15);
      return;
    }

    if (action == "speed_up") {
      move_speed_ = std::min(5.0, move_speed_ + 0.15);
      return;
    }

    if (action == "toggle_auto") {
      auto_mode_ = !auto_mode_;
      return;
    }
  }

  std::optional<int> pick_face(Vector2 mouse) const {
    Ray ray = GetMouseRay(mouse, camera_);

    float best_dist = std::numeric_limits<float>::infinity();
    int best_face = -1;

    for (int fi = 0; fi < sim_.mesh().n_faces(); ++fi) {
      const int i = sim_.mesh().faces(fi, 0);
      const int j = sim_.mesh().faces(fi, 1);
      const int k = sim_.mesh().faces(fi, 2);

      const Eigen::Vector3d vi = sim_.mesh().vertices.row(i);
      const Eigen::Vector3d vj = sim_.mesh().vertices.row(j);
      const Eigen::Vector3d vk = sim_.mesh().vertices.row(k);

      const Vector3 p1{static_cast<float>(vi.x()), static_cast<float>(vi.y()), static_cast<float>(vi.z())};
      const Vector3 p2{static_cast<float>(vj.x()), static_cast<float>(vj.y()), static_cast<float>(vj.z())};
      const Vector3 p3{static_cast<float>(vk.x()), static_cast<float>(vk.y()), static_cast<float>(vk.z())};

      const RayCollision hit = GetRayCollisionTriangle(ray, p1, p2, p3);
      if (hit.hit && hit.distance < best_dist) {
        best_dist = hit.distance;
        best_face = fi;
      }
    }

    if (best_face < 0) {
      return std::nullopt;
    }
    return best_face;
  }

  void handle_input() {
    const Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      if (const auto btn = button_at(mouse.x, mouse.y); btn.has_value()) {
        ui_drag_blocked_ = true;
        handle_action(*btn);
      } else if (is_in_panel(mouse.x, mouse.y)) {
        ui_drag_blocked_ = true;
      } else {
        if (const auto face = pick_face(mouse); face.has_value()) {
          selected_patch_ = sim_.patchification().face_labels(*face);
          update_planet_colors();
        }
      }
    }

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !ui_drag_blocked_) {
      dragging_ = true;
      const Vector2 d = GetMouseDelta();
      cam_yaw_ += d.x * 0.006F;
      cam_pitch_ += d.y * 0.006F;
      cam_pitch_ = std::clamp(cam_pitch_, -1.45F, 1.45F);
      rebuild_camera();
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
      dragging_ = false;
      ui_drag_blocked_ = false;
    }

    const float wheel = GetMouseWheelMove();
    if (std::abs(wheel) > 1e-6F) {
      cam_dist_ *= std::exp(-0.08F * wheel);
      cam_dist_ = std::clamp(cam_dist_, 1.4F, 10.0F);
      rebuild_camera();
    }

    if (IsKeyPressed(KEY_ESCAPE)) {
      CloseWindow();
    }
  }

  void update(float dt) {
    update_movement(dt);

    if (auto_mode_ && !moving_) {
      const int p = recommended_patch(true);
      selected_patch_ = p;
      queue_step(p);
    }
  }

  void draw_current_marker() const {
    const int cv = sim_.current_vertex();
    const Eigen::Vector3d p = sim_.mesh().vertices.row(cv);
    const Eigen::Vector3d n = mesh_normals_.row(cv);
    const double diag = (sim_.mesh().vertices.colwise().maxCoeff() - sim_.mesh().vertices.colwise().minCoeff()).norm();

    const Vector3 pos{static_cast<float>(p.x() + n.x() * 0.008 * diag),
                      static_cast<float>(p.y() + n.y() * 0.008 * diag),
                      static_cast<float>(p.z() + n.z() * 0.008 * diag)};

    DrawSphere(pos, 0.02F, YELLOW);
    DrawSphereWires(pos, 0.02F, 12, 12, BLACK);
  }

  void draw_last_path() const {
    if (sim_.history().empty()) {
      return;
    }

    const Eigen::VectorXi& path_vertices = sim_.history().back().path;
    const float outer_r = path_radius_;
    const Color path_color = Color{0, 220, 80, 255};

    for (Eigen::Index i = 0; i + 1 < path_vertices.size(); ++i) {
      const Eigen::Vector3d a = sim_.mesh().vertices.row(path_vertices(i));
      const Eigen::Vector3d b = sim_.mesh().vertices.row(path_vertices(i + 1));
      const Vector3 p0{static_cast<float>(a.x()), static_cast<float>(a.y()), static_cast<float>(a.z())};
      const Vector3 p1{static_cast<float>(b.x()), static_cast<float>(b.y()), static_cast<float>(b.z())};
      DrawCylinderEx(p0, p1, outer_r, outer_r, 12, path_color);
      DrawSphereEx(p0, outer_r * 1.05F, 8, 8, path_color);
    }
    const Eigen::Vector3d e = sim_.mesh().vertices.row(path_vertices(path_vertices.size() - 1));
    DrawSphereEx(Vector3{static_cast<float>(e.x()), static_cast<float>(e.y()), static_cast<float>(e.z())},
                 outer_r * 1.05F, 8, 8, path_color);
  }

  void draw_player() const {
    if (!has_player_) {
      return;
    }

    const Vector3 up = Vector3Normalize(player_normal_);
    const Vector3 pos = Vector3Add(player_position_,
                                   Vector3Scale(up, player_scale_ * player_base_height_ + player_clearance_));

    const Vector3 model_up{0.0F, 1.0F, 0.0F};
    Vector3 axis = Vector3CrossProduct(model_up, up);
    float axis_len = Vector3Length(axis);
    float angle = 0.0F;

    if (axis_len > 1e-6F) {
      axis = Vector3Scale(axis, 1.0F / axis_len);
      angle = std::acos(std::clamp(Vector3DotProduct(model_up, up), -1.0F, 1.0F)) * RAD2DEG;
    } else {
      axis = Vector3{1.0F, 0.0F, 0.0F};
      angle = (Vector3DotProduct(model_up, up) >= 0.0F) ? 0.0F : 180.0F;
    }

    DrawModelEx(player_model_, pos, axis, angle, Vector3{player_scale_, player_scale_, player_scale_}, WHITE);
  }

  void draw_scene() {
    DrawModel(planet_model_, Vector3{0.0F, 0.0F, 0.0F}, 1.0F, WHITE);
    draw_last_path();
    draw_player();
  }

  void draw() {
    BeginDrawing();
    ClearBackground(Color{18, 23, 34, 255});

    update_visual_uniforms();
    BeginMode3D(camera_);
    draw_scene();
    EndMode3D();

    draw_ui();

    EndDrawing();
  }

  SpectralBanditSimulator sim_;
  Eigen::MatrixXd mesh_normals_;

  bool show_panel_;
  int panel_width_;
  int ui_margin_;
  std::vector<UiButton> buttons_;

  Model planet_model_{};
  Texture2D planet_texture_{};

  bool has_player_;
  Model player_model_{};
  Texture2D player_texture_{};

  Camera3D camera_{};
  float cam_yaw_;
  float cam_pitch_;
  float cam_dist_;

  bool dragging_;
  bool ui_drag_blocked_;

  HeatmapMode mode_;
  int selected_patch_;

  bool auto_mode_;
  bool moving_;

  double avg_edge_len_;
  double move_speed_;
  float path_radius_;
  std::vector<int> move_path_;
  int move_progress_idx_;
  double move_progress_dist_;

  Vector3 player_position_{};
  Vector3 player_normal_{};
  Vector3 player_forward_{};
  float player_scale_;
  float player_base_height_;
  float player_clearance_;

  bool has_visual_shader_ = false;
  Shader visual_shader_{};
  int loc_light_pos_ = -1;
  int loc_light_color_ = -1;
  int loc_fill_light_dir_ = -1;
  int loc_fill_light_color_ = -1;
  int loc_view_pos_ = -1;
  int loc_ambient_color_ = -1;
  int loc_spec_power_ = -1;
  int loc_spec_strength_ = -1;
  int loc_rim_strength_ = -1;
  Vector3 light_pos_{};
  Vector3 light_color_{};
  Vector3 fill_light_dir_{};
  Vector3 fill_light_color_{};
  Vector3 ambient_color_{};
  float spec_power_ = 48.0F;
  float spec_strength_ = 0.45F;
  float rim_strength_ = 0.24F;

  std::string last_action_info_;
};

}  // namespace

Eigen::MatrixXd visibility_shadow_colors(const Eigen::VectorXd& values) {
  if (values.size() == 0) {
    return Eigen::MatrixXd(0, 3);
  }

  const double vmin = values.minCoeff();
  const double vmax = values.maxCoeff();

  Eigen::VectorXd vis(values.size());
  if (vmax - vmin < 1e-12) {
    vis.setConstant(0.01);
  } else {
    Eigen::VectorXd t = (values.array() - vmin) / (vmax - vmin);
    vis = 0.01 + 0.99 * t.array().pow(2.2);
  }

  Eigen::MatrixXd out(values.size(), 3);
  out.col(0) = vis;
  out.col(1) = vis;
  out.col(2) = vis;
  return out;
}

RenderMeshInfo load_render_mesh_info(const std::string& obj_path) {
  const ObjRenderData data = parse_obj_render(std::filesystem::path(obj_path));
  return RenderMeshInfo{.n_vertices = static_cast<int>(data.triangles.size() * 3),
                        .face_count = static_cast<int>(data.triangles.size())};
}

void run_game(const SimulatorConfig& config, const std::string& player_mesh_path, bool show_panel) {
  GameApp app(config, player_mesh_path, show_panel);
  app.run();
}

int render_heatmaps_png(const SimulatorConfig& config, int steps_before_render, const std::string& out_dir,
                        const std::string& player_mesh_path) {
  namespace fs = std::filesystem;
  fs::create_directories(out_dir);

  GameApp app(config, player_mesh_path, false);
  app.run_steps(steps_before_render);

  const std::vector<std::pair<HeatmapMode, std::string>> modes = {
      {HeatmapMode::Texture, "texture"},       {HeatmapMode::Exploration, "exploration"},
      {HeatmapMode::Exploitation, "exploitation"}, {HeatmapMode::UCB, "ucb"},
      {HeatmapMode::Oracle, "oracle"},         {HeatmapMode::Seen, "seen"},
  };

  for (const auto& [mode, name] : modes) {
    app.set_mode(mode);
    app.draw_once();

    Image img = LoadImageFromScreen();
    const fs::path out = fs::path(out_dir) / ("heatmap_" + name + ".png");
    ExportImage(img, out.string().c_str());
    UnloadImage(img);
  }

  return 0;
}

}  // namespace spectral_bandit

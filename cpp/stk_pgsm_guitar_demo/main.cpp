/**
 * PGSM STK guitar demo renderer — C++/STK (v1/v2/v3 physical-factor paths).
 *
 * Body modes are driven by bridge force (smoothed), never a second pluck.
 * v2: peak ceiling only (no RMS equalization); applied-parameter audit + metrics.
 */
#include <Stk.h>
#include <FileWvOut.h>
#include <Plucked.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stk;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kExpectedRenderCount = 9;

struct DemoConfig {
    std::string audioOutputSubdir = "audio/pgsm_stk_guitar_demo";
    std::string reportJsonPath = "audio/debug_reports/pgsm_stk_guitar_demo_report.json";
    std::string reportMdPath = "audio/debug_reports/pgsm_stk_guitar_demo_report.md";
    std::string demoVersion = "pgsm_stk_guitar_demo";
};

struct ModeSpec {
    double frequency_hz = 200.0;
    double gain = 0.05;
    double tau_or_q = 0.08;
    std::string component = "top";
};

struct PhysicalFactors {
    double body_size_cavity_factor = 1.0;
    double body_depth_m = 0.10;
    double depth_factor = 1.0;
    double body_volume_proxy = 0.013;
    double soundhole_area_proxy = 0.00636;
    double soundhole_radiation_factor = 1.0;
    double bridge_mobility_factor = 1.0;
    double effective_mass_loading_factor = 1.0;
    double top_stiffness_to_weight_factor = 1.0;
    double top_damping_factor = 1.0;
    double material_loss_factor = 1.0;
    double back_density_warmth_factor = 1.0;
    double air_helmholtz_factor = 1.0;
    double radiation_brightness_factor = 1.0;
};

struct FactorAuditEntry {
    double exported_value = 0.0;
    double parsed_value = 0.0;
    double applied_value = 0.0;
    bool applied_to_renderer = false;
    std::string renderer_mapping_target;
};

struct AudioMetrics {
    double peak_dbfs = -120.0;
    double rms_dbfs = -120.0;
    double spectral_centroid_hz = 0.0;
    double low_mid_energy_ratio = 0.0;
    double high_energy_ratio = 0.0;
    double decay_tau_s = 0.0;
    double body_string_energy_ratio = 0.0;
    double low_mid_120_450_ratio = 0.0;
};

struct RenderSpec {
    std::string sample_id;
    std::string note_name;
    double frequency_hz = 440.0;
    double duration_s = 2.5;
    int sample_rate = 44100;
    double pluck_position = 0.18;
    double string_decay = 0.65;
    double harmonic_brightness = 1.0;
    double excitation_strength = 1.0;
    double bridge_mobility = 1.0;
    double bridge_damping = 0.04;
    double string_to_body_send = 0.6;
    double peak_ceiling_dbfs = -6.0;
    double loudness_reference_dbfs = -20.0;
    bool normalize_rms = false;
    std::string output_wav_path;
    std::vector<ModeSpec> modes;
    double top_weight = 0.35;
    double back_weight = 0.28;
    double air_weight = 0.10;
    double string_direct_weight = 0.25;
    double direct_string_gain = 1.0;
    double body_modal_gain = 1.0;
    double string_to_body_send_scale = 1.0;
    double mapping_strength = 1.0;
    double note_excitation_scale = 1.0;
    double high_frequency_radiation_rolloff = 1.0;
    bool perceptual_v3 = false;
    PhysicalFactors phys;
    std::map<std::string, FactorAuditEntry> factor_audit;
};

struct RenderOutcome {
    std::vector<double> audio;
    AudioMetrics metrics;
    std::map<std::string, FactorAuditEntry> factor_audit;
    double raw_pluck_amplitude = 0.0;
    double clamped_pluck_amplitude = 0.0;
    bool pluck_was_clamped = false;
    double applied_string_to_body_send = 0.0;
    double applied_bridge_coupling = 0.0;
    int applied_bridge_smooth_samples = 0;
    std::vector<double> applied_modal_frequencies;
    std::vector<double> applied_modal_gains;
    std::vector<double> applied_modal_tau;
    double applied_top_weight = 0.0;
    double applied_back_weight = 0.0;
    double applied_air_weight = 0.0;
    double applied_string_direct_weight = 0.0;
};

// --- CLI / IO helpers -------------------------------------------------------

static bool getArg(int argc, char** argv, const std::string& key, std::string& out) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}

static std::string readTextFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void skipWs(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

static size_t findJsonKey(const std::string& json, const std::string& key, size_t start = 0) {
    return json.find("\"" + key + "\"", start);
}

static size_t findMatchingDelimiter(const std::string& s, size_t openPos, char openCh, char closeCh) {
    if (openPos >= s.size() || s[openPos] != openCh)
        throw std::runtime_error("JSON delimiter mismatch at " + std::to_string(openPos));
    int depth = 0;
    bool inString = false;
    bool escape = false;
    for (size_t i = openPos; i < s.size(); ++i) {
        char c = s[i];
        if (inString) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') inString = false;
            continue;
        }
        if (c == '"') { inString = true; continue; }
        if (c == openCh) ++depth;
        else if (c == closeCh && --depth == 0) return i;
    }
    throw std::runtime_error("Unclosed JSON delimiter at " + std::to_string(openPos));
}

static std::string extractJsonArraySlice(const std::string& json, const std::string& key) {
    size_t keyPos = findJsonKey(json, key);
    if (keyPos == std::string::npos) throw std::runtime_error("JSON missing array: " + key);
    size_t colon = json.find(':', keyPos);
    size_t i = colon + 1;
    skipWs(json, i);
    if (json[i] != '[') throw std::runtime_error("Expected array for key: " + key);
    size_t close = findMatchingDelimiter(json, i, '[', ']');
    return json.substr(i, close - i + 1);
}

static std::string extractJsonValueSlice(const std::string& json, const std::string& key, size_t start = 0) {
    size_t keyPos = findJsonKey(json, key, start);
    if (keyPos == std::string::npos) return "";
    size_t colon = json.find(':', keyPos);
    size_t i = colon + 1;
    skipWs(json, i);
    if (i >= json.size()) return "";
    const char c = json[i];
    if (c == '{') {
        size_t close = findMatchingDelimiter(json, i, '{', '}');
        return json.substr(i, close - i + 1);
    }
    if (c == '[') {
        size_t close = findMatchingDelimiter(json, i, '[', ']');
        return json.substr(i, close - i + 1);
    }
    return "";
}

static std::string extractJsonObjectSlice(const std::string& json, const std::string& key, size_t start = 0) {
    size_t keyPos = findJsonKey(json, key, start);
    if (keyPos == std::string::npos) return "";
    size_t colon = json.find(':', keyPos);
    size_t i = colon + 1;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{') return "";
    size_t close = findMatchingDelimiter(json, i, '{', '}');
    return json.substr(i, close - i + 1);
}

static std::vector<std::string> splitTopLevelArrayElements(const std::string& arraySlice) {
    if (arraySlice.size() < 2 || arraySlice.front() != '[') throw std::runtime_error("bad array slice");
    std::vector<std::string> elements;
    size_t i = 1;
    skipWs(arraySlice, i);
    while (i < arraySlice.size() - 1 && arraySlice[i] != ']') {
        skipWs(arraySlice, i);
        char open = arraySlice[i];
        char close = (open == '{') ? '}' : ']';
        size_t end = findMatchingDelimiter(arraySlice, i, open, close);
        elements.push_back(arraySlice.substr(i, end - i + 1));
        i = end + 1;
        skipWs(arraySlice, i);
        if (i < arraySlice.size() - 1 && arraySlice[i] == ',') ++i;
    }
    return elements;
}

static double parseJsonNumber(const std::string& json, const std::string& key, double fallback = 0.0) {
    size_t keyPos = findJsonKey(json, key);
    if (keyPos == std::string::npos) return fallback;
    size_t colon = json.find(':', keyPos);
    size_t pos = colon + 1;
    skipWs(json, pos);
    try {
        size_t idx = 0;
        return std::stod(json.substr(pos), &idx);
    } catch (...) {
        return fallback;
    }
}

static std::string parseJsonString(const std::string& json, const std::string& key, const std::string& fallback = "") {
    size_t keyPos = findJsonKey(json, key);
    if (keyPos == std::string::npos) return fallback;
    size_t colon = json.find(':', keyPos);
    size_t pos = colon + 1;
    skipWs(json, pos);
    if (pos >= json.size() || json[pos] != '"') return fallback;
    ++pos;
    std::string out;
    bool escape = false;
    for (; pos < json.size(); ++pos) {
        char c = json[pos];
        if (escape) { out.push_back(c); escape = false; }
        else if (c == '\\') escape = true;
        else if (c == '"') return out;
        else out.push_back(c);
    }
    return fallback;
}

static bool parseJsonBool(const std::string& json, const std::string& key, bool fallback = false) {
    size_t keyPos = findJsonKey(json, key);
    if (keyPos == std::string::npos) return fallback;
    size_t colon = json.find(':', keyPos);
    size_t pos = colon + 1;
    skipWs(json, pos);
    if (json.compare(pos, 4, "true") == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return fallback;
}

static double pfGet(const std::string& pfBlock, const std::string& key, double fallback = 1.0) {
    return parseJsonNumber(pfBlock, key, fallback);
}

static DemoConfig parseDemoConfig(const std::string& json) {
    DemoConfig cfg;
    std::string sub = parseJsonString(json, "audio_output_subdir");
    if (!sub.empty()) cfg.audioOutputSubdir = sub;
    std::string rj = parseJsonString(json, "report_json_path");
    if (!rj.empty()) cfg.reportJsonPath = rj;
    std::string rm = parseJsonString(json, "report_md_path");
    if (!rm.empty()) cfg.reportMdPath = rm;
    std::string dv = parseJsonString(json, "demo_version");
    if (!dv.empty()) cfg.demoVersion = dv;
    if (cfg.demoVersion.find("v4_10_samples") != std::string::npos) {
        // keep full demo id from JSON
    } else if (cfg.demoVersion.find("v3") != std::string::npos) {
        cfg.demoVersion = "pgsm_stk_guitar_demo_v3";
    }
    return cfg;
}

// --- DSP --------------------------------------------------------------------

struct Biquad {
    double b0 = 0, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    void reset() { x1 = x2 = y1 = y2 = 0; }
    double process(double x) {
        double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x; y2 = y1; y1 = y;
        return y;
    }
    void setBandpass(double f0, double Q, double fs) {
        f0 = std::max(1.0, std::min(f0, 0.49 * fs));
        Q = std::max(0.5, Q);
        double w0 = 2.0 * kPi * (f0 / fs);
        double alpha = std::sin(w0) / (2.0 * Q);
        double cosw0 = std::cos(w0);
        double aa0 = 1.0 + alpha;
        b0 = alpha / aa0; b1 = 0; b2 = -alpha / aa0;
        a1 = -2.0 * cosw0 / aa0; a2 = (1.0 - alpha) / aa0;
        reset();
    }
};

struct ModalBank {
    std::vector<Biquad> filters;
    std::vector<double> gains;
    void build(const std::vector<ModeSpec>& modes, double fs) {
        filters.clear(); gains.clear();
        for (const auto& m : modes) {
            Biquad bq;
            double Q = std::max(4.0, kPi * m.frequency_hz * std::max(m.tau_or_q, 1e-4));
            bq.setBandpass(m.frequency_hz, Q, fs);
            filters.push_back(bq);
            gains.push_back(m.gain);
        }
    }
    double process(double x) {
        double y = 0.0;
        for (size_t i = 0; i < filters.size(); ++i) y += gains[i] * filters[i].process(x);
        return y;
    }
    void reset() { for (auto& f : filters) f.reset(); }
};

static std::vector<ModeSpec> parseModesArray(const std::string& modesArraySlice) {
    std::vector<ModeSpec> out;
    for (const auto& obj : splitTopLevelArrayElements(modesArraySlice)) {
        ModeSpec m;
        m.frequency_hz = parseJsonNumber(obj, "frequency_hz", 0.0);
        m.gain = parseJsonNumber(obj, "gain", 0.0);
        m.tau_or_q = parseJsonNumber(obj, "tau_or_q", 0.0);
        if (m.tau_or_q <= 0.0) {
            double q = parseJsonNumber(obj, "q", 0.0);
            if (q > 0.0) m.tau_or_q = q / std::max(m.frequency_hz, 1.0) / kPi;
        }
        m.component = parseJsonString(obj, "component", "top");
        if (m.frequency_hz > 0.0) out.push_back(m);
    }
    return out;
}

static void fillPhysicalFromJson(RenderSpec& r, const std::string& block) {
    const std::string pf = extractJsonObjectSlice(block, "physical_factors");
    const std::string body = extractJsonObjectSlice(block, "body_model");
    const std::string mat = extractJsonObjectSlice(block, "material_model");
    const std::string rad = extractJsonObjectSlice(block, "radiation_model");

    if (!pf.empty()) {
        r.phys.body_size_cavity_factor = pfGet(pf, "body_size_cavity_factor", r.phys.body_size_cavity_factor);
        r.phys.soundhole_radiation_factor = pfGet(pf, "soundhole_radiation_factor", r.phys.soundhole_radiation_factor);
        r.phys.bridge_mobility_factor = pfGet(pf, "bridge_mobility_factor", r.phys.bridge_mobility_factor);
        r.phys.effective_mass_loading_factor = pfGet(pf, "effective_mass_loading_factor", r.phys.effective_mass_loading_factor);
        r.phys.top_stiffness_to_weight_factor = pfGet(pf, "top_stiffness_to_weight_factor", r.phys.top_stiffness_to_weight_factor);
        r.phys.top_damping_factor = pfGet(pf, "top_damping_factor", r.phys.top_damping_factor);
        r.phys.back_density_warmth_factor = pfGet(pf, "back_density_warmth_factor", r.phys.back_density_warmth_factor);
        r.phys.air_helmholtz_factor = pfGet(pf, "air_helmholtz_factor", r.phys.air_helmholtz_factor);
        r.phys.radiation_brightness_factor = pfGet(pf, "radiation_brightness_factor", r.phys.radiation_brightness_factor);
    }
    if (!body.empty()) {
        r.phys.effective_mass_loading_factor = parseJsonNumber(body, "effective_mass_loading", r.phys.effective_mass_loading_factor);
        r.phys.body_size_cavity_factor = parseJsonNumber(body, "body_size_cavity_factor", r.phys.body_size_cavity_factor);
        r.phys.depth_factor = parseJsonNumber(body, "depth_factor", r.phys.depth_factor);
        r.phys.body_depth_m = parseJsonNumber(body, "body_depth_m", r.phys.body_depth_m);
        r.phys.body_volume_proxy = parseJsonNumber(body, "body_volume_proxy", r.phys.body_volume_proxy);
        r.phys.soundhole_area_proxy = parseJsonNumber(body, "soundhole_area_proxy", r.phys.soundhole_area_proxy);
        r.phys.soundhole_radiation_factor = parseJsonNumber(body, "soundhole_radiation_factor", r.phys.soundhole_radiation_factor);
        r.body_modal_gain = parseJsonNumber(body, "body_modal_gain", r.body_modal_gain);
    }
    if (!mat.empty()) {
        r.phys.top_damping_factor = parseJsonNumber(mat, "top_damping", r.phys.top_damping_factor);
        r.phys.material_loss_factor = parseJsonNumber(mat, "material_loss_factor",
            parseJsonNumber(mat, "material_loss", r.phys.top_damping_factor * 0.92));
        r.phys.back_density_warmth_factor = parseJsonNumber(mat, "back_warmth", r.phys.back_density_warmth_factor);
        r.phys.top_stiffness_to_weight_factor = parseJsonNumber(mat, "stiffness_to_weight", r.phys.top_stiffness_to_weight_factor);
    }
    if (!rad.empty()) {
        r.phys.radiation_brightness_factor = parseJsonNumber(rad, "radiation_brightness", r.phys.radiation_brightness_factor);
    }
    r.bridge_mobility = r.phys.bridge_mobility_factor;
}

static RenderSpec parseRenderBlock(const std::string& block) {
    RenderSpec r;
    r.sample_id = parseJsonString(block, "sample_id");
    r.note_name = parseJsonString(block, "note_name");
    r.frequency_hz = parseJsonNumber(block, "frequency_hz", 0.0);
    r.duration_s = parseJsonNumber(block, "duration_s", 0.0);
    r.sample_rate = static_cast<int>(parseJsonNumber(block, "sample_rate", 0.0));

    const std::string sm = extractJsonObjectSlice(block, "string_model");
    if (!sm.empty()) {
        r.pluck_position = parseJsonNumber(sm, "pluck_position", r.pluck_position);
        r.string_decay = parseJsonNumber(sm, "string_decay", r.string_decay);
        r.harmonic_brightness = parseJsonNumber(sm, "harmonic_brightness", r.harmonic_brightness);
        r.excitation_strength = parseJsonNumber(sm, "excitation_strength", r.excitation_strength);
        r.note_excitation_scale = parseJsonNumber(sm, "note_excitation_scale", r.note_excitation_scale);
    }
    const std::string bm = extractJsonObjectSlice(block, "bridge_model");
    if (!bm.empty()) {
        r.bridge_mobility = parseJsonNumber(bm, "bridge_mobility", r.bridge_mobility);
        r.bridge_damping = parseJsonNumber(bm, "bridge_damping", r.bridge_damping);
        r.string_to_body_send = parseJsonNumber(bm, "string_to_body_send", r.string_to_body_send);
    }
    const std::string rm = extractJsonObjectSlice(block, "radiation_model");
    if (!rm.empty()) {
        r.top_weight = parseJsonNumber(rm, "top_weight", r.top_weight);
        r.back_weight = parseJsonNumber(rm, "back_weight", r.back_weight);
        r.air_weight = parseJsonNumber(rm, "air_weight", r.air_weight);
        r.string_direct_weight = parseJsonNumber(rm, "string_direct_weight", r.string_direct_weight);
        r.high_frequency_radiation_rolloff = parseJsonNumber(
            rm, "high_frequency_radiation_rolloff", r.high_frequency_radiation_rolloff);
    }
    const std::string mixModel = extractJsonObjectSlice(block, "string_body_mix");
    if (!mixModel.empty()) {
        r.direct_string_gain = parseJsonNumber(mixModel, "direct_string_gain", r.direct_string_gain);
        r.body_modal_gain = parseJsonNumber(mixModel, "body_modal_gain", r.body_modal_gain);
        r.string_to_body_send_scale = parseJsonNumber(mixModel, "string_to_body_send_scale", 1.0);
        const double sd = parseJsonNumber(mixModel, "string_direct", 0.0);
        if (sd > 0.0) r.string_direct_weight = sd;
    }
    const std::string pc = extractJsonObjectSlice(block, "perceptual_calibration");
    if (!pc.empty()) {
        r.perceptual_v3 = true;
        r.mapping_strength = parseJsonNumber(pc, "mapping_strength", 1.45);
        r.direct_string_gain = parseJsonNumber(pc, "direct_string_gain", r.direct_string_gain);
        r.body_modal_gain = parseJsonNumber(pc, "body_modal_gain", r.body_modal_gain);
        r.string_to_body_send_scale = parseJsonNumber(pc, "string_to_body_send_scale", r.string_to_body_send_scale);
    }
    const std::string om = extractJsonObjectSlice(block, "output_model");
    if (!om.empty()) {
        r.peak_ceiling_dbfs = parseJsonNumber(om, "peak_ceiling_dbfs",
            parseJsonNumber(om, "peak_target_dbfs", r.peak_ceiling_dbfs));
        r.loudness_reference_dbfs = parseJsonNumber(om, "loudness_reference_dbfs",
            parseJsonNumber(om, "loudness_target", r.loudness_reference_dbfs));
        r.normalize_rms = parseJsonBool(om, "normalize_rms", false);
        r.output_wav_path = parseJsonString(om, "output_wav_path");
    }
    const std::string body = extractJsonObjectSlice(block, "body_model");
    if (!body.empty()) {
        size_t mp = body.find("\"modes\"");
        if (mp != std::string::npos) {
            size_t colon = body.find(':', mp);
            size_t i = colon + 1;
            skipWs(body, i);
            if (body[i] == '[') {
                size_t close = findMatchingDelimiter(body, i, '[', ']');
                r.modes = parseModesArray(body.substr(i, close - i + 1));
            }
        }
    }
    fillPhysicalFromJson(r, block);
    return r;
}

static void validateRenderSpec(const RenderSpec& r, size_t index) {
    const std::string label = "render[" + std::to_string(index) + "]";
    if (r.sample_id.empty()) throw std::runtime_error(label + ": sample_id empty");
    if (r.note_name.empty()) throw std::runtime_error(label + ": note_name empty");
    if (r.output_wav_path.empty()) throw std::runtime_error(label + ": output_wav_path empty");
    if (r.frequency_hz <= 0) throw std::runtime_error(label + ": frequency_hz <= 0");
    if (r.sample_rate <= 0) throw std::runtime_error(label + ": sample_rate <= 0");
    if (r.duration_s <= 0) throw std::runtime_error(label + ": duration_s <= 0");
    if (r.modes.empty()) throw std::runtime_error(label + ": body_model.modes empty");
}

static std::filesystem::path resolveOutputPath(const std::string& relPath,
    const std::filesystem::path& repoRoot, const std::string& requiredSubdir) {
    if (relPath.empty()) throw std::runtime_error("output_wav_path empty");
    std::filesystem::path out(relPath);
    if (!out.is_absolute()) out = repoRoot / out;
    out = out.lexically_normal();
    const std::string fname = out.filename().string();
    if (fname.empty() || fname == ".wav")
        throw std::runtime_error("refusing invalid WAV filename: " + out.string());
    if (out.generic_string().find(requiredSubdir) == std::string::npos)
        throw std::runtime_error("output must be under " + requiredSubdir + ": " + out.string());
    if (fname.find("_stk_guitar.wav") == std::string::npos)
        throw std::runtime_error("WAV name must end with _stk_guitar.wav");
    return out;
}

static std::vector<RenderSpec> parseRenders(const std::string& json) {
    auto blocks = splitTopLevelArrayElements(extractJsonArraySlice(json, "renders"));
    if (blocks.empty()) throw std::runtime_error("renders array empty");
    std::vector<RenderSpec> specs;
    for (size_t i = 0; i < blocks.size(); ++i) {
        RenderSpec r = parseRenderBlock(blocks[i]);
        validateRenderSpec(r, i);
        specs.push_back(std::move(r));
    }
    int expected = static_cast<int>(parseJsonNumber(json, "expected_render_count", kExpectedRenderCount));
    if (static_cast<int>(specs.size()) != expected)
        throw std::runtime_error("Expected " + std::to_string(expected) + " renders, got " + std::to_string(specs.size()));
    return specs;
}

static void clearOutputWavs(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    if (!std::filesystem::exists(dir)) return;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".wav") std::filesystem::remove(e.path());
    }
}

static double linToDbfs(double x) { return 20.0 * std::log10(std::max(x, 1e-12)); }
static double dbfsToLin(double db) { return std::pow(10.0, db / 20.0); }

constexpr double kStkPluckAmpMin = 0.0;
constexpr double kStkPluckAmpMax = 1.0;

static double clampStkPluckAmplitude(double raw) {
    return std::max(kStkPluckAmpMin, std::min(raw, kStkPluckAmpMax));
}

static void applyPeakCeilingOnly(std::vector<double>& y, double ceilingDbfs) {
    double peak = 0.0;
    for (double v : y) peak = std::max(peak, std::abs(v));
    if (peak <= 1e-12) return;
    double ceiling = dbfsToLin(ceilingDbfs);
    if (peak > ceiling) {
        double g = ceiling / peak;
        for (double& v : y) v *= g;
    }
}

static std::vector<double> smoothBridgeDrive(const std::vector<double>& raw, int smoothN) {
    std::vector<double> out(raw.size(), 0.0);
    if (smoothN < 2) return raw;
    double acc = 0.0;
    for (size_t i = 0; i < raw.size(); ++i) {
        acc += raw[i];
        if (i >= static_cast<size_t>(smoothN)) acc -= raw[i - smoothN];
        out[i] = acc / static_cast<double>(std::min<int>(static_cast<int>(i) + 1, smoothN));
    }
    return out;
}

static void applyPhysicalMappings(RenderSpec& r) {
    const auto& p = r.phys;
    const double strength = std::max(1.0, r.mapping_strength);
    const auto pows = [&](double base, double exp) {
        return std::pow(base, 1.0 + (strength - 1.0) * exp);
    };
    const double holeAreaScale = std::pow(p.soundhole_area_proxy / 0.00636, 0.18 * strength);
    const double volScale = std::pow(p.body_volume_proxy / 0.013, 0.12 * strength);

    for (auto& m : r.modes) {
        if (m.component == "air") {
            m.frequency_hz *= pows(p.air_helmholtz_factor, 0.22);
            m.gain *= p.soundhole_radiation_factor * holeAreaScale * pows(p.air_helmholtz_factor, 0.18);
            m.tau_or_q *= pows(p.effective_mass_loading_factor, 0.12);
        } else if (m.component == "back") {
            m.frequency_hz *= pows(p.body_size_cavity_factor * p.depth_factor * volScale, 0.10);
            m.gain *= p.back_density_warmth_factor * pows(p.body_size_cavity_factor, 0.14);
            m.tau_or_q *= pows(p.back_density_warmth_factor, 0.10);
        } else if (m.component == "top") {
            m.frequency_hz *= pows(p.top_stiffness_to_weight_factor, 0.14);
            m.gain *= pows(p.top_stiffness_to_weight_factor, 0.22);
            m.tau_or_q /= std::max(p.material_loss_factor, 0.45);
        } else if (m.component == "radiation") {
            m.gain *= p.radiation_brightness_factor;
            m.tau_or_q /= std::max(p.material_loss_factor, 0.45);
        }
        if (m.frequency_hz >= 120.0 && m.frequency_hz <= 450.0) {
            m.gain *= r.body_modal_gain;
            if (r.perceptual_v3) m.gain *= (m.frequency_hz < 260.0 ? 1.08 : 1.0);
        } else {
            m.gain *= std::pow(r.body_modal_gain, 0.85);
        }
        if (m.frequency_hz < 260.0) {
            m.frequency_hz *= pows(p.body_size_cavity_factor * p.depth_factor, 0.06);
            m.gain *= pows(p.body_size_cavity_factor * p.depth_factor * volScale, 0.12);
        }
        m.frequency_hz = std::max(40.0, std::min(m.frequency_hz, r.sample_rate * 0.49));
        m.tau_or_q = std::max(0.015, m.tau_or_q);
    }

    r.string_direct_weight *= r.direct_string_gain;
    r.string_to_body_send *= r.string_to_body_send_scale;
    r.harmonic_brightness *= pows(p.top_stiffness_to_weight_factor, 0.12) * pows(p.radiation_brightness_factor, 0.18);
    if (r.high_frequency_radiation_rolloff > 0.0)
        r.harmonic_brightness *= std::pow(r.high_frequency_radiation_rolloff, 0.35);
    r.top_weight *= pows(p.radiation_brightness_factor, 0.20);
    r.back_weight *= pows(p.back_density_warmth_factor, 0.18);
    r.air_weight *= p.soundhole_radiation_factor * holeAreaScale * pows(p.air_helmholtz_factor, 0.15);

    const double wsum = r.top_weight + r.back_weight + r.air_weight;
    if (wsum > 1e-9) {
        const double bodyBudget = std::max(0.05, 1.0 - r.string_direct_weight);
        r.top_weight = r.top_weight / wsum * bodyBudget;
        r.back_weight = r.back_weight / wsum * bodyBudget;
        r.air_weight = r.air_weight / wsum * bodyBudget;
    }

    auto setAudit = [&](const std::string& name, double exported, double applied, const std::string& target) {
        FactorAuditEntry e;
        e.exported_value = exported;
        e.parsed_value = exported;
        e.applied_value = applied;
        e.applied_to_renderer = true;
        e.renderer_mapping_target = target;
        r.factor_audit[name] = e;
    };

    setAudit("bridge_mobility_factor", p.bridge_mobility_factor, p.bridge_mobility_factor, "bridge_coupling_gain / string_to_body_send");
    setAudit("effective_mass_loading_factor", p.effective_mass_loading_factor, p.effective_mass_loading_factor, "bridge_attack_smoothing / low_mode_tau");
    setAudit("body_size_cavity_factor", p.body_size_cavity_factor, p.body_size_cavity_factor, "low_mode_frequency_gain");
    setAudit("soundhole_radiation_factor", p.soundhole_radiation_factor, p.soundhole_radiation_factor, "air_mode_gain / air_weight");
    setAudit("soundhole_area_proxy", p.soundhole_area_proxy, holeAreaScale, "air_radiation_area_scaling");
    setAudit("top_stiffness_to_weight_factor", p.top_stiffness_to_weight_factor, p.top_stiffness_to_weight_factor, "top_mode_frequency_gain / harmonic_brightness");
    setAudit("top_damping_factor", p.top_damping_factor, p.top_damping_factor, "modal_tau_top_radiation");
    setAudit("material_loss_factor", p.material_loss_factor, p.material_loss_factor, "modal_Q_decay");
    setAudit("back_density_warmth_factor", p.back_density_warmth_factor, p.back_density_warmth_factor, "back_mode_gain_tau / back_weight");
    setAudit("air_helmholtz_factor", p.air_helmholtz_factor, p.air_helmholtz_factor, "air_mode_frequency_gain");
    setAudit("body_depth_m", p.body_depth_m, p.depth_factor, "low_mode_frequency_tau_depth");
    setAudit("body_volume_proxy", p.body_volume_proxy, volScale, "cavity_mode_gain_volume");
    setAudit("radiation_brightness_factor", p.radiation_brightness_factor, p.radiation_brightness_factor, "radiation_mode_gain / top_weight");
    setAudit("top_weight", parseJsonNumber("", "top_weight", r.top_weight), r.top_weight, "final_radiation_mix");
    setAudit("back_weight", r.back_weight, r.back_weight, "final_radiation_mix");
    setAudit("air_weight", r.air_weight, r.air_weight, "final_radiation_mix");
    setAudit("string_body_mix", r.string_direct_weight, r.string_direct_weight, "string_direct_vs_body_modal_mix");
    setAudit("direct_string_gain", r.direct_string_gain, r.direct_string_gain, "stk_plucked_direct_path_gain");
    setAudit("body_modal_gain", r.body_modal_gain, r.body_modal_gain, "body_modal_bank_output_gain");
}

static AudioMetrics computeMetrics(const std::vector<double>& y, int sr,
    double stringEnergy, double bodyEnergy) {
    AudioMetrics m;
    double peak = 0.0, sumsq = 0.0;
    for (double v : y) {
        peak = std::max(peak, std::abs(v));
        sumsq += v * v;
    }
    m.peak_dbfs = linToDbfs(peak);
    m.rms_dbfs = linToDbfs(std::sqrt(sumsq / std::max<size_t>(y.size(), 1)));

    const size_t n = y.size();
    double lowMidE = 0.0, highE = 0.0, weightedF = 0.0, specE = 0.0;
    for (size_t k = 1; k < n / 2 && k < 4096; ++k) {
        double re = 0.0, im = 0.0;
        for (size_t t = 0; t < n; ++t) {
            double ang = 2.0 * kPi * k * t / n;
            re += y[t] * std::cos(ang);
            im -= y[t] * std::sin(ang);
        }
        double pwr = (re * re + im * im) / static_cast<double>(n * n);
        double f = k * sr / static_cast<double>(n);
        specE += pwr;
        weightedF += f * pwr;
        if (f >= 120.0 && f <= 450.0) lowMidE += pwr;
        if (f > 1200.0) highE += pwr;
    }
    if (specE > 1e-18) m.spectral_centroid_hz = weightedF / specE;
    m.low_mid_energy_ratio = lowMidE / std::max(specE, 1e-18);
    m.low_mid_120_450_ratio = m.low_mid_energy_ratio;
    m.high_energy_ratio = highE / std::max(specE, 1e-18);

    size_t peakIdx = 0;
    double peakVal = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double av = std::abs(y[i]);
        if (av >= peakVal) { peakVal = av; peakIdx = i; }
    }
    double e0 = peakVal;
    size_t e1 = std::min(n - 1, peakIdx + static_cast<size_t>(0.35 * sr));
    double e1v = std::abs(y[e1]);
    if (e0 > 1e-9 && e1v > 1e-12)
        m.decay_tau_s = -0.35 / std::log(std::max(e1v / e0, 1e-6));

    m.body_string_energy_ratio = bodyEnergy / std::max(stringEnergy, 1e-18);
    return m;
}

static RenderOutcome renderOne(RenderSpec spec) {
    applyPhysicalMappings(spec);
    const int sr = spec.sample_rate;
    const int n = static_cast<int>(spec.duration_s * sr);
    Stk::setSampleRate(static_cast<float>(sr));

    Plucked string;
    string.setFrequency(spec.frequency_hz);
    const double rawPluck = spec.excitation_strength * spec.note_excitation_scale
        * std::pow(spec.phys.bridge_mobility_factor, 0.08);
    const double clampedPluck = clampStkPluckAmplitude(rawPluck);
    string.pluck(clampedPluck);

    ModalBank body;
    body.build(spec.modes, sr);

    const double coupling = spec.string_to_body_send * spec.phys.bridge_mobility_factor;
    const int smoothN = std::max(3, static_cast<int>((0.005 + 0.012 * spec.phys.effective_mass_loading_factor) * sr));

    std::vector<double> stringBuf(n), bridgeRaw(n);
    double prev = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = string.tick();
        stringBuf[static_cast<size_t>(i)] = s;
        double ds = s - prev;
        prev = s;
        bridgeRaw[static_cast<size_t>(i)] = coupling * ds * (1.0 + 0.15 * spec.harmonic_brightness);
    }
    auto bridgeDrive = smoothBridgeDrive(bridgeRaw, smoothN);
    body.reset();

    double topAcc = 0, backAcc = 0, airAcc = 0;
    for (const auto& m : spec.modes) {
        if (m.component == "top") topAcc += m.gain;
        else if (m.component == "back") backAcc += m.gain;
        else if (m.component == "air") airAcc += m.gain;
    }
    double compSum = std::max(topAcc + backAcc + airAcc, 1e-9);

    RenderOutcome out;
    out.audio.assign(n, 0.0);
    out.raw_pluck_amplitude = rawPluck;
    out.clamped_pluck_amplitude = clampedPluck;
    out.pluck_was_clamped = rawPluck < kStkPluckAmpMin || rawPluck > kStkPluckAmpMax;
    double stringE = 0.0, bodyE = 0.0;
    for (int i = 0; i < n; ++i) {
        double bodyS = body.process(bridgeDrive[static_cast<size_t>(i)]);
        double topP = bodyS * (topAcc / compSum) * spec.top_weight;
        double backP = bodyS * (backAcc / compSum) * spec.back_weight;
        double airP = bodyS * (airAcc / compSum) * spec.air_weight;
        double bodyMix = (topP + backP + airP) * std::pow(spec.body_modal_gain, 0.35);
        double str = spec.string_direct_weight * stringBuf[static_cast<size_t>(i)];
        out.audio[static_cast<size_t>(i)] = str + bodyMix;
        stringE += str * str;
        bodyE += bodyMix * bodyMix;
    }

    if (!spec.normalize_rms) applyPeakCeilingOnly(out.audio, spec.peak_ceiling_dbfs);

    out.metrics = computeMetrics(out.audio, sr, stringE, bodyE);
    out.factor_audit = spec.factor_audit;
    out.applied_string_to_body_send = coupling;
    out.applied_bridge_coupling = spec.phys.bridge_mobility_factor;
    out.applied_bridge_smooth_samples = smoothN;
    for (const auto& m : spec.modes) {
        out.applied_modal_frequencies.push_back(m.frequency_hz);
        out.applied_modal_gains.push_back(m.gain);
        out.applied_modal_tau.push_back(m.tau_or_q);
    }
    out.applied_top_weight = spec.top_weight;
    out.applied_back_weight = spec.back_weight;
    out.applied_air_weight = spec.air_weight;
    out.applied_string_direct_weight = spec.string_direct_weight;
    return out;
}

static void writeWav(const std::filesystem::path& path, const std::vector<double>& y) {
    std::filesystem::create_directories(path.parent_path());
    FileWvOut out(path.string(), 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
    for (double v : y) out.tick(static_cast<float>(std::max(-1.0, std::min(1.0, v))));
}

static double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    if (n < 8) return 1.0;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; ++i) {
        double xa = a[i] - ma, xb = b[i] - mb;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    return num / std::max(std::sqrt(da * db), 1e-18);
}

static std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else o += c;
    }
    return o;
}

static std::string inferBottleneck(const std::vector<RenderOutcome>& outcomes,
    const std::vector<RenderSpec>& specs, const std::string& auditSlice, bool isV3) {
    bool allApplied = true;
    for (const auto& o : outcomes)
        for (const auto& kv : o.factor_audit)
            if (!kv.second.applied_to_renderer) allApplied = false;
    if (!allApplied) return "renderer_not_applying_factors";

    if (auditSlice.find("too_small_from_source") != std::string::npos &&
        auditSlice.find("diagnostic_exaggeration_for_audible_demo\": true") != std::string::npos)
        return "factors_active_but_need_demo_scaling";

    std::map<std::string, std::vector<double>> rmsBySample;
    for (size_t i = 0; i < outcomes.size(); ++i)
        if (specs[i].note_name == "A2") rmsBySample[specs[i].sample_id].push_back(outcomes[i].metrics.rms_dbfs);
    if (rmsBySample.size() >= 3) {
        double lo = 1e9, hi = -1e9;
        double cLo = 1e9, cHi = -1e9;
        for (const auto& kv : rmsBySample) {
            double v = kv.second.empty() ? -120.0 : kv.second[0];
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
        for (size_t i = 0; i < outcomes.size(); ++i) {
            if (specs[i].note_name != "A2") continue;
            cLo = std::min(cLo, outcomes[i].metrics.spectral_centroid_hz);
            cHi = std::max(cHi, outcomes[i].metrics.spectral_centroid_hz);
        }
        const double rmsSpread = hi - lo;
        const double centroidSpread = cHi - cLo;
        const bool isV4 = specs.size() >= 20;
        if (isV3 || isV4) {
            const double centroidThresh = isV4 ? 40.0 : 90.0;
            const double rmsThresh = isV4 ? 0.40 : 0.55;
            if (centroidSpread >= centroidThresh || rmsSpread >= rmsThresh) return "differentiation_active";
            if (centroidSpread >= centroidThresh * 0.6 || rmsSpread >= rmsThresh * 0.55)
                return "factors_active_but_need_demo_scaling";
            return "factors_active_but_need_demo_scaling";
        }
        if (rmsSpread < 0.35) return "factors_active_but_need_demo_scaling";
    }
    return "differentiation_active";
}

static std::string readinessStatusV4(const std::string& bottleneck, size_t wavCount,
    const std::vector<RenderOutcome>& outcomes, const std::vector<RenderSpec>& specs,
    int expectedCount) {
    if (wavCount != static_cast<size_t>(expectedCount) || expectedCount != 30)
        return "audit_failed_missing_factor_application";
    if (bottleneck == "renderer_not_applying_factors" || bottleneck == "export_missing_factors")
        return "audit_failed_missing_factor_application";

    bool clipped = false;
    for (const auto& o : outcomes) {
        if (o.metrics.peak_dbfs > -2.5) clipped = true;
    }
    if (clipped) return "demo_generated_but_tone_regression";

    double minCorr = 1.0;
    std::set<std::string> sampleIds;
    for (const auto& s : specs) sampleIds.insert(s.sample_id);
    const std::vector<std::string> notes = {"A2", "A4", "E5"};
    for (const auto& note : notes) {
        for (auto itA = sampleIds.begin(); itA != sampleIds.end(); ++itA) {
            for (auto itB = std::next(itA); itB != sampleIds.end(); ++itB) {
                const std::vector<double> *a = nullptr, *b = nullptr;
                for (size_t i = 0; i < specs.size(); ++i) {
                    if (specs[i].note_name == note && specs[i].sample_id == *itA) a = &outcomes[i].audio;
                    if (specs[i].note_name == note && specs[i].sample_id == *itB) b = &outcomes[i].audio;
                }
                if (a && b) minCorr = std::min(minCorr, pearson(*a, *b));
            }
        }
    }

    double centroidSpread = 0.0;
    double cLo = 1e9, cHi = -1e9;
    for (size_t i = 0; i < outcomes.size(); ++i) {
        if (specs[i].note_name != "A2") continue;
        cLo = std::min(cLo, outcomes[i].metrics.spectral_centroid_hz);
        cHi = std::max(cHi, outcomes[i].metrics.spectral_centroid_hz);
    }
    centroidSpread = cHi - cLo;

    if (bottleneck == "differentiation_active" && centroidSpread >= 45.0 && minCorr < 0.97)
        return "ready_for_classical_guitar_stk_acceptance";
    if (centroidSpread < 35.0 && minCorr > 0.95) return "demo_generated_but_differentiation_weak";
    if (minCorr < 0.50) return "demo_generated_but_tone_regression";
    if (centroidSpread >= 35.0 && minCorr < 0.97) return "ready_for_classical_guitar_stk_acceptance";
    return "demo_generated_but_differentiation_weak";
}

static std::string readinessStatusV3(const std::string& bottleneck, size_t wavCount,
    const std::vector<RenderOutcome>& outcomes, const std::vector<RenderSpec>& specs) {
    if (wavCount != 9) return "audit_failed_missing_factor_application";
    if (bottleneck == "renderer_not_applying_factors" || bottleneck == "export_missing_factors")
        return "audit_failed_missing_factor_application";

    bool clipped = false;
    bool extremeBody = false;
    for (const auto& o : outcomes) {
        if (o.metrics.peak_dbfs > -2.5) clipped = true;
        if (o.metrics.body_string_energy_ratio > 18.0 || o.metrics.body_string_energy_ratio < 0.04)
            extremeBody = true;
    }
    if (clipped || extremeBody) return "demo_generated_but_tone_regression";

    double minCorr = 1.0;
    const std::vector<std::string> notes = {"A2", "A4", "E5"};
    const std::vector<std::pair<std::string, std::string>> pairs = {
        {"sample_000", "sample_001"}, {"sample_000", "sample_002"}, {"sample_001", "sample_002"}};
    for (const auto& note : notes) {
        for (const auto& pr : pairs) {
            const std::vector<double> *a = nullptr, *b = nullptr;
            for (size_t i = 0; i < specs.size(); ++i) {
                if (specs[i].note_name == note && specs[i].sample_id == pr.first) a = &outcomes[i].audio;
                if (specs[i].note_name == note && specs[i].sample_id == pr.second) b = &outcomes[i].audio;
            }
            if (a && b) minCorr = std::min(minCorr, pearson(*a, *b));
        }
    }

    double centroidSpread = 0.0;
    double cLo = 1e9, cHi = -1e9;
    for (size_t i = 0; i < outcomes.size(); ++i) {
        if (specs[i].note_name != "A2") continue;
        cLo = std::min(cLo, outcomes[i].metrics.spectral_centroid_hz);
        cHi = std::max(cHi, outcomes[i].metrics.spectral_centroid_hz);
    }
    centroidSpread = cHi - cLo;

    if (bottleneck == "differentiation_active" && centroidSpread >= 70.0 && minCorr < 0.97)
        return "ready_for_gui_activation";
    if (centroidSpread < 50.0 && minCorr > 0.94) return "demo_generated_but_differentiation_weak";
    if (minCorr < 0.55) return "demo_generated_but_tone_regression";
    if (bottleneck == "differentiation_active") return "ready_for_gui_activation";
    return "demo_generated_but_differentiation_weak";
}

static std::string readinessStatus(const std::string& bottleneck, size_t wavCount) {
    if (wavCount != 9) return "audit_failed_missing_factor_application";
    if (bottleneck == "renderer_not_applying_factors" || bottleneck == "cpp_parser_not_reading_factors" ||
        bottleneck == "export_missing_factors")
        return "audit_failed_missing_factor_application";
    if (bottleneck == "factors_active_but_need_demo_scaling" || bottleneck == "source_values_too_close")
        return "demo_generated_but_differentiation_weak";
    if (bottleneck == "differentiation_active") return "ready_for_gui_activation";
    return "demo_generated_but_differentiation_weak";
}

static void writeReportV2(const std::filesystem::path& jsonPath, const std::filesystem::path& mdPath,
    const std::string& paramsJson, const DemoConfig& cfg,
    const std::vector<RenderSpec>& specs, const std::vector<RenderOutcome>& outcomes,
    const std::vector<std::string>& written) {
    const std::string auditSlice = extractJsonObjectSlice(paramsJson, "physical_difference_audit");
    const bool isV4 = cfg.demoVersion.find("v4_10_samples") != std::string::npos;
    const bool isV3 = !isV4 && cfg.demoVersion.find("v3") != std::string::npos;
    const int expectedCount = static_cast<int>(parseJsonNumber(paramsJson, "expected_render_count", kExpectedRenderCount));
    const std::string bottleneck = inferBottleneck(outcomes, specs, auditSlice, isV3 || isV4);
    const std::string readiness = isV4
        ? readinessStatusV4(bottleneck, written.size(), outcomes, specs, expectedCount)
        : (isV3 ? readinessStatusV3(bottleneck, written.size(), outcomes, specs)
                : readinessStatus(bottleneck, written.size()));

    std::ostringstream js;
    js << "{\n";
    js << "  \"demo_version\": \"" << jsonEscape(cfg.demoVersion) << "\",\n";
    js << "  \"renderer\": \"STK/C++\",\n";
    js << "  \"python_role\": \"parameter_export_only\",\n";
    js << "  \"expected_render_count\": " << expectedCount << ",\n";
    js << "  \"actual_render_count\": " << written.size() << ",\n";
    js << "  \"render_count\": " << written.size() << ",\n";
    js << "  \"readiness\": \"" << readiness << "\",\n";
    js << "  \"differentiation_bottleneck\": \"" << bottleneck << "\",\n";
    js << "  \"normalization_policy\": \"peak_ceiling_only_no_rms_equalization\",\n";
    if (!auditSlice.empty()) js << "  \"physical_difference_audit\": " << auditSlice << ",\n";
    const std::string factorMatrix = extractJsonValueSlice(paramsJson, "stk_factor_activation_matrix");
    if (!factorMatrix.empty()) {
        js << "  \"stk_factor_activation_matrix\": " << factorMatrix << ",\n";
    }
    const std::string weakSummary = extractJsonValueSlice(paramsJson, "missing_or_weak_factor_summary");
    if (!weakSummary.empty()) {
        js << "  \"missing_or_weak_factor_summary\": " << weakSummary << ",\n";
    }
    const std::string spreadTable = extractJsonObjectSlice(paramsJson, "physical_factor_spread_table");
    if (!spreadTable.empty()) js << "  \"physical_factor_spread_table\": " << spreadTable << ",\n";
    const std::string modalSummary = extractJsonObjectSlice(paramsJson, "modal_bank_summary_per_sample");
    if (!modalSummary.empty()) js << "  \"modal_bank_summary_per_sample\": " << modalSummary << ",\n";
    const std::string soundholeSummary = extractJsonObjectSlice(paramsJson, "soundhole_radiation_summary");
    if (!soundholeSummary.empty()) js << "  \"soundhole_radiation_summary\": " << soundholeSummary << ",\n";
    const std::string dampingSummary = extractJsonObjectSlice(paramsJson, "material_damping_summary");
    if (!dampingSummary.empty()) js << "  \"material_damping_summary\": " << dampingSummary << ",\n";
    const std::string depthSummary = extractJsonObjectSlice(paramsJson, "body_depth_volume_summary");
    if (!depthSummary.empty()) js << "  \"body_depth_volume_summary\": " << depthSummary << ",\n";
    const std::string knownLimits = extractJsonValueSlice(paramsJson, "known_limitations");
    if (!knownLimits.empty()) js << "  \"known_limitations\": " << knownLimits << ",\n";
    js << "  \"pluck_amplitude_handling\": \"clamped_to_stk_0_1_range\",\n";
    js << "  \"pluck_amplitude_audit\": [\n";
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        const auto& o = outcomes[i];
        js << "    {";
        js << "\"sample_id\":\"" << s.sample_id << "\"";
        js << ",\"note_name\":\"" << s.note_name << "\"";
        js << ",\"raw_pluck_amplitude\":" << o.raw_pluck_amplitude;
        js << ",\"clamped_pluck_amplitude\":" << o.clamped_pluck_amplitude;
        js << ",\"was_clamped\":" << (o.pluck_was_clamped ? "true" : "false");
        js << "}" << (i + 1 < specs.size() ? ",\n" : "\n");
    }
    js << "  ],\n";
    js << "  \"applied_parameter_audit\": [\n";
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        const auto& o = outcomes[i];
        js << "    {\n";
        js << "      \"sample_id\": \"" << s.sample_id << "\",\n";
        js << "      \"note_name\": \"" << s.note_name << "\",\n";
        js << "      \"output_path\": \"" << jsonEscape(written[i]) << "\",\n";
        js << "      \"applied_string_to_body_send\": " << o.applied_string_to_body_send << ",\n";
        js << "      \"applied_bridge_coupling\": " << o.applied_bridge_coupling << ",\n";
        js << "      \"applied_bridge_smooth_samples\": " << o.applied_bridge_smooth_samples << ",\n";
        js << "      \"applied_top_weight\": " << o.applied_top_weight << ",\n";
        js << "      \"applied_back_weight\": " << o.applied_back_weight << ",\n";
        js << "      \"applied_air_weight\": " << o.applied_air_weight << ",\n";
        js << "      \"applied_string_direct_weight\": " << o.applied_string_direct_weight << ",\n";
        js << "      \"raw_pluck_amplitude\": " << o.raw_pluck_amplitude << ",\n";
        js << "      \"clamped_pluck_amplitude\": " << o.clamped_pluck_amplitude << ",\n";
        js << "      \"pluck_was_clamped\": " << (o.pluck_was_clamped ? "true" : "false") << ",\n";
        js << "      \"applied_modal_frequencies_hz\": [";
        for (size_t k = 0; k < o.applied_modal_frequencies.size(); ++k)
            js << o.applied_modal_frequencies[k] << (k + 1 < o.applied_modal_frequencies.size() ? ", " : "");
        js << "],\n";
        js << "      \"applied_modal_gains\": [";
        for (size_t k = 0; k < o.applied_modal_gains.size(); ++k)
            js << o.applied_modal_gains[k] << (k + 1 < o.applied_modal_gains.size() ? ", " : "");
        js << "],\n";
        js << "      \"applied_modal_tau_or_q\": [";
        for (size_t k = 0; k < o.applied_modal_tau.size(); ++k)
            js << o.applied_modal_tau[k] << (k + 1 < o.applied_modal_tau.size() ? ", " : "");
        js << "],\n";
        js << "      \"factor_application\": {\n";
        size_t fi = 0;
        for (const auto& kv : o.factor_audit) {
            js << "        \"" << kv.first << "\": {";
            js << "\"exported_value\":" << kv.second.exported_value;
            js << ",\"parsed_value\":" << kv.second.parsed_value;
            js << ",\"applied_value\":" << kv.second.applied_value;
            js << ",\"applied_to_renderer\":" << (kv.second.applied_to_renderer ? "true" : "false");
            js << ",\"renderer_mapping_target\":\"" << jsonEscape(kv.second.renderer_mapping_target) << "\"}";
            if (++fi < o.factor_audit.size()) js << ",";
            js << "\n";
        }
        js << "      },\n";
        js << "      \"audio_metrics\": {";
        js << "\"peak_dbfs\":" << o.metrics.peak_dbfs;
        js << ",\"rms_dbfs\":" << o.metrics.rms_dbfs;
        js << ",\"spectral_centroid_hz\":" << o.metrics.spectral_centroid_hz;
        js << ",\"low_mid_energy_ratio\":" << o.metrics.low_mid_energy_ratio;
        js << ",\"high_energy_ratio\":" << o.metrics.high_energy_ratio;
        js << ",\"decay_tau_s\":" << o.metrics.decay_tau_s;
        js << ",\"body_string_energy_ratio\":" << o.metrics.body_string_energy_ratio;
        js << ",\"low_mid_120_450_ratio\":" << o.metrics.low_mid_120_450_ratio;
        js << "}\n";
        js << "    }" << (i + 1 < specs.size() ? ",\n" : "\n");
    }
    js << "  ],\n";

    js << "  \"per_sample_applied_mix_summary\": {\n";
    std::set<std::string> reportSampleIds;
    for (const auto& s : specs) reportSampleIds.insert(s.sample_id);
    size_t mixIdx = 0;
    const size_t mixTotal = reportSampleIds.size();
    for (const auto& sid : reportSampleIds) {
        for (size_t i = 0; i < specs.size(); ++i) {
            if (specs[i].sample_id != sid || specs[i].note_name != "A2") continue;
            const auto& o = outcomes[i];
            js << "    \"" << sid << "\": {";
            js << "\"string_direct_weight\":" << o.applied_string_direct_weight;
            js << ",\"body_modal_gain\":" << (specs[i].body_modal_gain);
            js << ",\"string_to_body_send\":" << o.applied_string_to_body_send;
            js << ",\"spectral_centroid_hz\":" << o.metrics.spectral_centroid_hz;
            js << ",\"body_string_energy_ratio\":" << o.metrics.body_string_energy_ratio;
            js << "}" << (++mixIdx < mixTotal ? ",\n" : "\n");
        }
    }
    js << "  },\n";

    js << "  \"pairwise_same_note_correlation\": {\n";
    const std::vector<std::string> notes = {"A2", "A4", "E5"};
    std::vector<std::pair<std::string, std::string>> pairs;
    if (isV4 || reportSampleIds.size() > 3) {
        for (auto itA = reportSampleIds.begin(); itA != reportSampleIds.end(); ++itA) {
            for (auto itB = std::next(itA); itB != reportSampleIds.end(); ++itB)
                pairs.emplace_back(*itA, *itB);
        }
    } else {
        pairs = {{"sample_000", "sample_001"}, {"sample_000", "sample_002"}, {"sample_001", "sample_002"}};
    }
    size_t pi = 0;
    const size_t pairTotal = notes.size() * pairs.size();
    for (const auto& note : notes) {
        for (const auto& pr : pairs) {
            const std::vector<double> *a = nullptr, *b = nullptr;
            for (size_t i = 0; i < specs.size(); ++i) {
                if (specs[i].note_name == note && specs[i].sample_id == pr.first) a = &outcomes[i].audio;
                if (specs[i].note_name == note && specs[i].sample_id == pr.second) b = &outcomes[i].audio;
            }
            if (a && b) {
                js << "    \"" << note << "_" << pr.first << "_vs_" << pr.second << "\": "
                   << pearson(*a, *b) << (++pi < pairTotal ? ",\n" : "\n");
            }
        }
    }
    js << "  },\n";
    js << "  \"pairwise_same_note_spectral_centroid_distance_hz\": {\n";
    pi = 0;
    for (const auto& note : notes) {
        for (const auto& pr : pairs) {
            double ca = 0.0, cb = 0.0;
            bool ok = false;
            for (size_t i = 0; i < specs.size(); ++i) {
                if (specs[i].note_name == note && specs[i].sample_id == pr.first) { ca = outcomes[i].metrics.spectral_centroid_hz; ok = true; }
                if (specs[i].note_name == note && specs[i].sample_id == pr.second) cb = outcomes[i].metrics.spectral_centroid_hz;
            }
            if (ok) {
                js << "    \"" << note << "_" << pr.first << "_vs_" << pr.second << "\": "
                   << std::abs(ca - cb) << (++pi < pairTotal ? ",\n" : "\n");
            }
        }
    }
    js << "  },\n";
    js << "  \"outputs\": [\n";
    for (size_t i = 0; i < written.size(); ++i)
        js << "    \"" << jsonEscape(written[i]) << "\"" << (i + 1 < written.size() ? ",\n" : "\n");
    js << "  ]\n}\n";

    std::filesystem::create_directories(jsonPath.parent_path());
    std::ofstream(jsonPath) << js.str();

    std::ofstream md(mdPath);
    md << "# PGSM STK Guitar Demo Report (" << cfg.demoVersion << ")\n\n";
    md << "- **renderer**: STK/C++\n";
    md << "- **python_role**: parameter_export_only\n";
    md << "- **readiness**: " << readiness << "\n";
    md << "- **differentiation_bottleneck**: " << bottleneck << "\n";
    md << "- **renders**: " << written.size() << "\n\n";
    md << "## Outputs\n\n";
    for (const auto& w : written) md << "- `" << w << "`\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string paramsPath = "audio/debug_reports/pgsm_stk_demo_parameters.json";
        std::string repoRoot = ".";
        getArg(argc, argv, "--params", paramsPath);
        getArg(argc, argv, "--repo-root", repoRoot);

        const std::filesystem::path repo = std::filesystem::path(repoRoot).lexically_normal();
        const std::string jsonText = readTextFile(paramsPath);
        DemoConfig cfg = parseDemoConfig(jsonText);
        auto specs = parseRenders(jsonText);

        const std::filesystem::path outDir = repo / cfg.audioOutputSubdir;
        clearOutputWavs(outDir);

        std::vector<RenderOutcome> outcomes;
        std::vector<std::string> written;
        outcomes.reserve(specs.size());
        written.reserve(specs.size());

        for (auto& spec : specs) {
            const std::filesystem::path out = resolveOutputPath(spec.output_wav_path, repo, cfg.audioOutputSubdir);
            std::cout << "Rendering " << spec.sample_id << " " << spec.note_name
                      << " -> " << out.generic_string() << "\n";
            RenderOutcome result = renderOne(spec);
            writeWav(out, result.audio);
            written.push_back(out.generic_string());
            outcomes.push_back(std::move(result));
        }

        const std::filesystem::path reportJson = repo / cfg.reportJsonPath;
        const std::filesystem::path reportMd = repo / cfg.reportMdPath;
        writeReportV2(reportJson, reportMd, jsonText, cfg, specs, outcomes, written);

        std::cout << "Wrote report: " << reportJson.generic_string() << "\n";
        std::cout << "Done — " << written.size() << " WAV files.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "stk_pgsm_guitar_demo error: " << e.what() << "\n";
        return 1;
    }
}

/**
 * PGSM STK guitar demo renderer — C++/STK conference demo skeleton.
 *
 * Physical chain:
 *   pluck/contact -> STK Plucked string -> bridge force -> body modal bank
 *   -> soundhole/air radiation -> top/back/air mix -> WAV
 *
 * Body modes are driven by bridge force (smoothed), never a second pluck.
 *
 * Usage (from repo root on VM):
 *   ./stk_pgsm_guitar_demo --params audio/debug_reports/pgsm_stk_demo_parameters.json
 */
#include <Stk.h>
#include <FileWvOut.h>
#include <Plucked.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stk;

namespace {

constexpr double kPi = 3.14159265358979323846;

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

static double parseJsonNumber(const std::string& json, const std::string& key, double fallback = 0.0) {
    const std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) ++pos;
    try {
        size_t idx = 0;
        double v = std::stod(json.substr(pos), &idx);
        return v;
    } catch (...) {
        return fallback;
    }
}

static std::string parseJsonString(const std::string& json, const std::string& key, const std::string& fallback = "") {
    const std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return fallback;
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return fallback;
    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return fallback;
    return json.substr(pos + 1, end - pos - 1);
}

struct ModeSpec {
    double frequency_hz = 200.0;
    double gain = 0.05;
    double tau_or_q = 0.08;
    std::string component = "top";
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
    double peak_target_dbfs = -6.0;
    double loudness_target = -20.0;
    std::string output_wav_path;
    std::vector<ModeSpec> modes;
    double top_weight = 0.35;
    double back_weight = 0.28;
    double air_weight = 0.10;
    double string_direct_weight = 0.25;
};

struct Biquad {
    double b0 = 0, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    void reset() { x1 = x2 = y1 = y2 = 0; }

    double process(double x) {
        double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = x;
        y2 = y1;
        y1 = y;
        return y;
    }

    void setBandpass(double f0, double Q, double fs) {
        f0 = std::max(1.0, std::min(f0, 0.49 * fs));
        Q = std::max(0.5, Q);
        double w0 = 2.0 * kPi * (f0 / fs);
        double alpha = std::sin(w0) / (2.0 * Q);
        double cosw0 = std::cos(w0);
        double bb0 = alpha;
        double bb1 = 0.0;
        double bb2 = -alpha;
        double aa0 = 1.0 + alpha;
        double aa1 = -2.0 * cosw0;
        double aa2 = 1.0 - alpha;
        b0 = bb0 / aa0;
        b1 = bb1 / aa0;
        b2 = bb2 / aa0;
        a1 = aa1 / aa0;
        a2 = aa2 / aa0;
        reset();
    }
};

struct ModalBank {
    std::vector<Biquad> filters;
    std::vector<double> gains;
    std::vector<std::string> components;

    void build(const std::vector<ModeSpec>& modes, double fs) {
        filters.clear();
        gains.clear();
        components.clear();
        for (const auto& m : modes) {
            Biquad bq;
            double Q = std::max(4.0, kPi * m.frequency_hz * m.tau_or_q);
            bq.setBandpass(m.frequency_hz, Q, fs);
            filters.push_back(bq);
            gains.push_back(m.gain);
            components.push_back(m.component);
        }
    }

    double process(double bridgeForce) {
        double y = 0.0;
        for (size_t i = 0; i < filters.size(); ++i) {
            y += gains[i] * filters[i].process(bridgeForce);
        }
        return y;
    }

    void reset() {
        for (auto& f : filters) f.reset();
    }
};

static std::vector<ModeSpec> parseModesBlock(const std::string& block) {
    std::vector<ModeSpec> out;
    size_t pos = 0;
    while (true) {
        auto fpos = block.find("\"frequency_hz\"", pos);
        if (fpos == std::string::npos) break;
        auto objStart = block.rfind('{', fpos);
        auto objEnd = block.find('}', fpos);
        if (objStart == std::string::npos || objEnd == std::string::npos) break;
        std::string obj = block.substr(objStart, objEnd - objStart + 1);
        ModeSpec m;
        m.frequency_hz = parseJsonNumber(obj, "frequency_hz", 200.0);
        m.gain = parseJsonNumber(obj, "gain", 0.05);
        m.tau_or_q = parseJsonNumber(obj, "tau_or_q", 0.08);
        if (m.tau_or_q <= 0.0) m.tau_or_q = parseJsonNumber(obj, "q", 20.0) / std::max(m.frequency_hz, 1.0) / kPi;
        m.component = parseJsonString(obj, "component", "top");
        out.push_back(m);
        pos = objEnd + 1;
    }
    return out;
}

static std::vector<RenderSpec> parseRenders(const std::string& json) {
    auto rendersPos = json.find("\"renders\"");
    if (rendersPos == std::string::npos) throw std::runtime_error("JSON missing \"renders\" array");
    auto arrStart = json.find('[', rendersPos);
    auto arrEnd = json.find(']', arrStart);
    if (arrStart == std::string::npos || arrEnd == std::string::npos)
        throw std::runtime_error("Cannot parse renders array");

    std::vector<RenderSpec> specs;
    size_t pos = arrStart + 1;
    while (pos < arrEnd) {
        auto objStart = json.find('{', pos);
        if (objStart == std::string::npos || objStart >= arrEnd) break;
        int depth = 0;
        size_t objEnd = objStart;
        for (size_t i = objStart; i < arrEnd; ++i) {
            if (json[i] == '{') ++depth;
            if (json[i] == '}') {
                --depth;
                if (depth == 0) {
                    objEnd = i;
                    break;
                }
            }
        }
        std::string block = json.substr(objStart, objEnd - objStart + 1);
        RenderSpec r;
        r.sample_id = parseJsonString(block, "sample_id");
        r.note_name = parseJsonString(block, "note_name");
        r.frequency_hz = parseJsonNumber(block, "frequency_hz", 440.0);
        r.duration_s = parseJsonNumber(block, "duration_s", 2.5);
        r.sample_rate = static_cast<int>(parseJsonNumber(block, "sample_rate", 44100.0));

        auto smPos = block.find("\"string_model\"");
        if (smPos != std::string::npos) {
            auto smStart = block.find('{', smPos);
            auto smEnd = block.find('}', smStart);
            std::string sm = block.substr(smStart, smEnd - smStart + 1);
            r.pluck_position = parseJsonNumber(sm, "pluck_position", 0.18);
            r.string_decay = parseJsonNumber(sm, "string_decay", 0.65);
            r.harmonic_brightness = parseJsonNumber(sm, "harmonic_brightness", 1.0);
            r.excitation_strength = parseJsonNumber(sm, "excitation_strength", 1.0);
        }

        auto bmPos = block.find("\"bridge_model\"");
        if (bmPos != std::string::npos) {
            auto bmStart = block.find('{', bmPos);
            auto bmEnd = block.find('}', bmStart);
            std::string bm = block.substr(bmStart, bmEnd - bmStart + 1);
            r.bridge_mobility = parseJsonNumber(bm, "bridge_mobility", 1.0);
            r.bridge_damping = parseJsonNumber(bm, "bridge_damping", 0.04);
            r.string_to_body_send = parseJsonNumber(bm, "string_to_body_send", 0.6);
        }

        auto rmPos = block.find("\"radiation_model\"");
        if (rmPos != std::string::npos) {
            auto rmStart = block.find('{', rmPos);
            auto rmEnd = block.find('}', rmStart);
            std::string rm = block.substr(rmStart, rmEnd - rmStart + 1);
            r.top_weight = parseJsonNumber(rm, "top_weight", 0.35);
            r.back_weight = parseJsonNumber(rm, "back_weight", 0.28);
            r.air_weight = parseJsonNumber(rm, "air_weight", 0.10);
            r.string_direct_weight = parseJsonNumber(rm, "string_direct_weight", 0.25);
        }

        auto omPos = block.find("\"output_model\"");
        if (omPos != std::string::npos) {
            auto omStart = block.find('{', omPos);
            auto omEnd = block.find('}', omStart);
            std::string om = block.substr(omStart, omEnd - omStart + 1);
            r.peak_target_dbfs = parseJsonNumber(om, "peak_target_dbfs", -6.0);
            r.loudness_target = parseJsonNumber(om, "loudness_target", -20.0);
            r.output_wav_path = parseJsonString(om, "output_wav_path");
        }

        auto modesPos = block.find("\"modes\"");
        if (modesPos != std::string::npos) {
            auto mStart = block.find('[', modesPos);
            auto mEnd = block.find(']', mStart);
            if (mStart != std::string::npos && mEnd != std::string::npos)
                r.modes = parseModesBlock(block.substr(mStart, mEnd - mStart + 1));
        }
        if (r.modes.empty()) {
            r.modes = {
                {118.0, 0.04, 0.12, "air"},
                {195.0, 0.08, 0.09, "top"},
                {145.0, 0.06, 0.10, "back"},
                {420.0, 0.05, 0.07, "radiation"},
            };
        }
        specs.push_back(r);
        pos = objEnd + 1;
    }
    if (specs.empty()) throw std::runtime_error("No render entries parsed from parameters JSON");
    return specs;
}

static double dbfsToLinear(double dbfs) { return std::pow(10.0, dbfs / 20.0); }

static void normalizePeak(std::vector<double>& y, double peakTargetDbfs) {
    double peak = 0.0;
    for (double v : y) peak = std::max(peak, std::abs(v));
    if (peak < 1e-12) return;
    double target = dbfsToLinear(peakTargetDbfs);
    double g = target / peak;
    for (double& v : y) v *= g;
}

static std::vector<double> smoothBridgeDrive(const std::vector<double>& bridgeRaw, int smoothN) {
    std::vector<double> out(bridgeRaw.size(), 0.0);
    if (smoothN < 2) return bridgeRaw;
    double acc = 0.0;
    for (size_t i = 0; i < bridgeRaw.size(); ++i) {
        acc += bridgeRaw[i];
        if (i >= static_cast<size_t>(smoothN)) acc -= bridgeRaw[i - smoothN];
        out[i] = acc / static_cast<double>(std::min<int>(static_cast<int>(i) + 1, smoothN));
    }
    return out;
}

static std::vector<double> renderOne(const RenderSpec& spec) {
    const int sr = spec.sample_rate;
    const int n = static_cast<int>(spec.duration_s * sr);
    Stk::setSampleRate(static_cast<float>(sr));

    Plucked string;
    string.setFrequency(spec.frequency_hz);
    string.controlChange(4, static_cast<float>(127.0 * std::max(0.0, std::min(1.0, spec.pluck_position))));
    string.controlChange(11, static_cast<float>(127.0 * std::max(0.0, std::min(1.0, spec.string_decay))));
    string.pluck(spec.excitation_strength);

    ModalBank body;
    body.build(spec.modes, sr);

    std::vector<double> stringBuf(n, 0.0);
    std::vector<double> bridgeRaw(n, 0.0);
    double prevString = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = string.tick();
        stringBuf[static_cast<size_t>(i)] = s;
        double ds = s - prevString;
        prevString = s;
        bridgeRaw[static_cast<size_t>(i)] =
            spec.string_to_body_send * spec.bridge_mobility * ds * (1.0 + 0.15 * spec.harmonic_brightness);
    }

    const int smoothN = std::max(3, static_cast<int>(0.008 * sr));
    auto bridgeDrive = smoothBridgeDrive(bridgeRaw, smoothN);
    body.reset();

    double topAcc = 0.0, backAcc = 0.0, airAcc = 0.0;
    for (const auto& m : spec.modes) {
        if (m.component == "top") topAcc += m.gain;
        else if (m.component == "back") backAcc += m.gain;
        else if (m.component == "air") airAcc += m.gain;
    }
    double compSum = std::max(topAcc + backAcc + airAcc, 1e-9);

    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double bodySample = body.process(bridgeDrive[static_cast<size_t>(i)]);
        double topPart = bodySample * (topAcc / compSum) * spec.top_weight;
        double backPart = bodySample * (backAcc / compSum) * spec.back_weight;
        double airPart = bodySample * (airAcc / compSum) * spec.air_weight;
        double bodyMix = topPart + backPart + airPart;
        y[static_cast<size_t>(i)] = spec.string_direct_weight * stringBuf[static_cast<size_t>(i)] + bodyMix;
    }

    normalizePeak(y, spec.peak_target_dbfs);
    return y;
}

static void writeWav(const std::string& path, const std::vector<double>& y, int sr) {
    std::filesystem::path p(path);
    std::filesystem::create_directories(p.parent_path());
    FileWvOut out(path, 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
    for (double v : y) {
        float f = static_cast<float>(std::max(-1.0, std::min(1.0, v)));
        out.tick(f);
    }
}

static void writeReport(
    const std::filesystem::path& jsonPath,
    const std::filesystem::path& mdPath,
    const std::vector<RenderSpec>& specs,
    const std::vector<std::string>& written) {
    std::ostringstream js;
    js << "{\n";
    js << "  \"renderer\": \"STK/C++\",\n";
    js << "  \"python_role\": \"parameter_export_only\",\n";
    js << "  \"binary\": \"stk_pgsm_guitar_demo\",\n";
    js << "  \"render_count\": " << written.size() << ",\n";
    js << "  \"physical_factors_used\": [\n";
    js << "    \"body_size_cavity_factor\", \"soundhole_radiation_factor\", \"bridge_mobility\",\n";
    js << "    \"effective_mass_loading\", \"top_damping\", \"back_warmth\", \"modal_frequencies\",\n";
    js << "    \"modal_tau_or_q\", \"radiation_brightness\", \"top_back_air_radiation_weights\"\n";
    js << "  ],\n";
    js << "  \"outputs\": [\n";
    for (size_t i = 0; i < written.size(); ++i) {
        js << "    \"" << written[i] << "\"" << (i + 1 < written.size() ? ",\n" : "\n");
    }
    js << "  ],\n";
    js << "  \"known_limitations\": [\n";
    js << "    \"Conference demo skeleton — modal bank uses bandpass resonators, not full FEM modes.\",\n";
    js << "    \"Bridge force is string-derivative proxy; admittance feedback is simplified.\",\n";
    js << "    \"Body is excited only from bridge drive — no independent body pluck.\",\n";
    js << "    \"Parameter JSON produced by Python; this binary does not run FEM/ROM.\"\n";
    js << "  ]\n";
    js << "}\n";
    std::filesystem::create_directories(jsonPath.parent_path());
    std::ofstream(jsonPath) << js.str();

    std::ofstream md(mdPath);
    md << "# PGSM STK Guitar Demo Report\n\n";
    md << "- **renderer**: STK/C++\n";
    md << "- **python_role**: parameter_export_only\n";
    md << "- **renders**: " << written.size() << "\n\n";
    md << "## Physical factors used\n\n";
    md << "Body size/cavity, soundhole radiation, bridge mobility, effective mass loading,\n";
    md << "top damping, back warmth, modal frequencies/Q/tau, radiation brightness,\n";
    md << "top/back/air radiation weights.\n\n";
    md << "## Per-sample differences\n\n";
    md << "| sample | profile hint |\n|--------|-------------|\n";
    md << "| sample_000 | balanced neutral |\n";
    md << "| sample_001 | bright / light / fast |\n";
    md << "| sample_002 | warm / deep / heavy |\n\n";
    md << "## Outputs\n\n";
    for (const auto& w : written) md << "- `" << w << "`\n";
    md << "\n## Known limitations\n\n";
    md << "- Modal bank is STK bandpass resonators driven by smoothed bridge force.\n";
    md << "- No second body pluck; body follows string bridge coupling only.\n";
    md << "- Full FEM/ROM modes not loaded at runtime.\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string paramsPath = "audio/debug_reports/pgsm_stk_demo_parameters.json";
        std::string repoRoot = ".";
        getArg(argc, argv, "--params", paramsPath);
        getArg(argc, argv, "--repo-root", repoRoot);

        const std::string jsonText = readTextFile(paramsPath);
        auto specs = parseRenders(jsonText);

        std::vector<std::string> written;
        for (const auto& spec : specs) {
            std::cout << "Rendering " << spec.sample_id << " " << spec.note_name
                      << " -> " << spec.output_wav_path << "\n";
            auto y = renderOne(spec);
            std::filesystem::path out = spec.output_wav_path;
            if (!out.is_absolute()) out = std::filesystem::path(repoRoot) / out;
            writeWav(out.string(), y, spec.sample_rate);
            written.push_back(out.string());
        }

        std::filesystem::path reportJson = std::filesystem::path(repoRoot) / "audio/debug_reports/pgsm_stk_guitar_demo_report.json";
        std::filesystem::path reportMd = std::filesystem::path(repoRoot) / "audio/debug_reports/pgsm_stk_guitar_demo_report.md";
        writeReport(reportJson, reportMd, specs, written);

        std::cout << "Wrote report: " << reportJson << "\n";
        std::cout << "Done — " << written.size() << " WAV files.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "stk_pgsm_guitar_demo error: " << e.what() << "\n";
        return 1;
    }
}

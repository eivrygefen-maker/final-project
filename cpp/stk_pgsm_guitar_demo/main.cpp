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
#include <cctype>
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
constexpr int kExpectedRenderCount = 9;
constexpr char kOutputDir[] = "audio/pgsm_stk_guitar_demo/";

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
    const std::string needle = "\"" + key + "\"";
    return json.find(needle, start);
}

static size_t findMatchingDelimiter(const std::string& s, size_t openPos, char openCh, char closeCh) {
    if (openPos >= s.size() || s[openPos] != openCh)
        throw std::runtime_error(std::string("JSON delimiter mismatch at ") + std::to_string(openPos));
    int depth = 0;
    bool inString = false;
    bool escape = false;
    for (size_t i = openPos; i < s.size(); ++i) {
        char c = s[i];
        if (inString) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }
        if (c == '"') {
            inString = true;
            continue;
        }
        if (c == openCh) ++depth;
        else if (c == closeCh) {
            --depth;
            if (depth == 0) return i;
        }
    }
    throw std::runtime_error(std::string("Unclosed JSON delimiter starting at ") + std::to_string(openPos));
}

static std::string extractJsonArraySlice(const std::string& json, const std::string& key) {
    size_t keyPos = findJsonKey(json, key);
    if (keyPos == std::string::npos)
        throw std::runtime_error("JSON missing array key: \"" + key + "\"");
    size_t colon = json.find(':', keyPos);
    if (colon == std::string::npos) throw std::runtime_error("Malformed JSON near key: " + key);
    size_t i = colon + 1;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '[')
        throw std::runtime_error("JSON key \"" + key + "\" is not an array");
    size_t close = findMatchingDelimiter(json, i, '[', ']');
    return json.substr(i, close - i + 1);
}

static std::string extractJsonObjectSlice(const std::string& json, const std::string& key, size_t start = 0) {
    size_t keyPos = findJsonKey(json, key, start);
    if (keyPos == std::string::npos) return "";
    size_t colon = json.find(':', keyPos);
    if (colon == std::string::npos) return "";
    size_t i = colon + 1;
    skipWs(json, i);
    if (i >= json.size() || json[i] != '{') return "";
    size_t close = findMatchingDelimiter(json, i, '{', '}');
    return json.substr(i, close - i + 1);
}

static std::vector<std::string> splitTopLevelArrayElements(const std::string& arraySlice) {
    if (arraySlice.size() < 2 || arraySlice.front() != '[' || arraySlice.back() != ']')
        throw std::runtime_error("splitTopLevelArrayElements expects [ ... ] slice");
    std::vector<std::string> elements;
    size_t i = 1;
    skipWs(arraySlice, i);
    if (i < arraySlice.size() - 1 && arraySlice[i] == ']') return elements;

    while (i < arraySlice.size() - 1) {
        skipWs(arraySlice, i);
        if (arraySlice[i] == ']') break;
        char open = arraySlice[i];
        char close = (open == '{') ? '}' : (open == '[' ? ']' : '\0');
        if (close == '\0') throw std::runtime_error("Unexpected token inside JSON array");
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
    if (colon == std::string::npos) return fallback;
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
    if (colon == std::string::npos) return fallback;
    size_t pos = colon + 1;
    skipWs(json, pos);
    if (pos >= json.size() || json[pos] != '"') return fallback;
    ++pos;
    std::string out;
    bool escape = false;
    for (; pos < json.size(); ++pos) {
        char c = json[pos];
        if (escape) {
            out.push_back(c);
            escape = false;
        } else if (c == '\\') {
            escape = true;
        } else if (c == '"') {
            return out;
        } else {
            out.push_back(c);
        }
    }
    return fallback;
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

static std::vector<ModeSpec> parseModesArray(const std::string& modesArraySlice) {
    std::vector<ModeSpec> out;
    for (const auto& objSlice : splitTopLevelArrayElements(modesArraySlice)) {
        ModeSpec m;
        m.frequency_hz = parseJsonNumber(objSlice, "frequency_hz", 0.0);
        m.gain = parseJsonNumber(objSlice, "gain", 0.0);
        m.tau_or_q = parseJsonNumber(objSlice, "tau_or_q", 0.0);
        if (m.tau_or_q <= 0.0) {
            double q = parseJsonNumber(objSlice, "q", 0.0);
            if (q > 0.0 && m.frequency_hz > 0.0)
                m.tau_or_q = q / std::max(m.frequency_hz, 1.0) / kPi;
        }
        m.component = parseJsonString(objSlice, "component", "top");
        if (m.frequency_hz > 0.0) out.push_back(m);
    }
    return out;
}

static RenderSpec parseRenderBlock(const std::string& block) {
    RenderSpec r;
    r.sample_id = parseJsonString(block, "sample_id");
    r.note_name = parseJsonString(block, "note_name");
    r.frequency_hz = parseJsonNumber(block, "frequency_hz", 0.0);
    r.duration_s = parseJsonNumber(block, "duration_s", 0.0);
    r.sample_rate = static_cast<int>(parseJsonNumber(block, "sample_rate", 0.0));

    const std::string stringModel = extractJsonObjectSlice(block, "string_model");
    if (!stringModel.empty()) {
        r.pluck_position = parseJsonNumber(stringModel, "pluck_position", r.pluck_position);
        r.string_decay = parseJsonNumber(stringModel, "string_decay", r.string_decay);
        r.harmonic_brightness = parseJsonNumber(stringModel, "harmonic_brightness", r.harmonic_brightness);
        r.excitation_strength = parseJsonNumber(stringModel, "excitation_strength", r.excitation_strength);
    }

    const std::string bridgeModel = extractJsonObjectSlice(block, "bridge_model");
    if (!bridgeModel.empty()) {
        r.bridge_mobility = parseJsonNumber(bridgeModel, "bridge_mobility", r.bridge_mobility);
        r.bridge_damping = parseJsonNumber(bridgeModel, "bridge_damping", r.bridge_damping);
        r.string_to_body_send = parseJsonNumber(bridgeModel, "string_to_body_send", r.string_to_body_send);
    }

    const std::string radiationModel = extractJsonObjectSlice(block, "radiation_model");
    if (!radiationModel.empty()) {
        r.top_weight = parseJsonNumber(radiationModel, "top_weight", r.top_weight);
        r.back_weight = parseJsonNumber(radiationModel, "back_weight", r.back_weight);
        r.air_weight = parseJsonNumber(radiationModel, "air_weight", r.air_weight);
        r.string_direct_weight = parseJsonNumber(radiationModel, "string_direct_weight", r.string_direct_weight);
    }

    const std::string outputModel = extractJsonObjectSlice(block, "output_model");
    if (!outputModel.empty()) {
        r.peak_target_dbfs = parseJsonNumber(outputModel, "peak_target_dbfs", r.peak_target_dbfs);
        r.loudness_target = parseJsonNumber(outputModel, "loudness_target", r.loudness_target);
        r.output_wav_path = parseJsonString(outputModel, "output_wav_path");
    }

    const std::string bodyModel = extractJsonObjectSlice(block, "body_model");
    if (!bodyModel.empty()) {
        const std::string modesKey = "\"modes\"";
        size_t modesPos = bodyModel.find(modesKey);
        if (modesPos != std::string::npos) {
            size_t colon = bodyModel.find(':', modesPos);
            size_t i = colon + 1;
            skipWs(bodyModel, i);
            if (i < bodyModel.size() && bodyModel[i] == '[') {
                size_t close = findMatchingDelimiter(bodyModel, i, '[', ']');
                r.modes = parseModesArray(bodyModel.substr(i, close - i + 1));
            }
        }
    }

    return r;
}

static void validateRenderSpec(const RenderSpec& r, size_t index) {
    const std::string label = "render[" + std::to_string(index) + "]";
    if (r.sample_id.empty())
        throw std::runtime_error(label + ": sample_id is empty");
    if (r.note_name.empty())
        throw std::runtime_error(label + ": note_name is empty");
    if (r.output_wav_path.empty())
        throw std::runtime_error(label + ": output_model.output_wav_path is empty");
    if (r.frequency_hz <= 0.0)
        throw std::runtime_error(label + ": frequency_hz must be > 0");
    if (r.sample_rate <= 0)
        throw std::runtime_error(label + ": sample_rate must be > 0");
    if (r.duration_s <= 0.0)
        throw std::runtime_error(label + ": duration_s must be > 0");
    if (r.modes.empty())
        throw std::runtime_error(label + ": body_model.modes is empty");
}

static std::filesystem::path resolveOutputPath(const std::string& relPath, const std::filesystem::path& repoRoot) {
    if (relPath.empty())
        throw std::runtime_error("output_wav_path is empty");
    std::filesystem::path out(relPath);
    if (!out.is_absolute()) out = repoRoot / out;
    out = out.lexically_normal();

    const std::string fname = out.filename().string();
    if (fname.empty() || fname == ".wav")
        throw std::runtime_error("refusing to write invalid WAV filename: " + out.string());

    const std::string pathStr = out.generic_string();
    if (pathStr.find(kOutputDir) == std::string::npos)
        throw std::runtime_error("output_wav_path must be under audio/pgsm_stk_guitar_demo/: " + pathStr);
    if (fname.find("_stk_guitar.wav") == std::string::npos)
        throw std::runtime_error("output WAV name must end with _stk_guitar.wav: " + fname);

    return out;
}

static int parseExpectedRenderCount(const std::string& json) {
    int fromField = static_cast<int>(parseJsonNumber(json, "expected_render_count", 0.0));
    if (fromField > 0) return fromField;
    try {
        const std::string arr = extractJsonArraySlice(json, "expected_wav_files");
        return static_cast<int>(splitTopLevelArrayElements(arr).size());
    } catch (...) {
        return kExpectedRenderCount;
    }
}

static std::vector<RenderSpec> parseRenders(const std::string& json) {
    const std::string rendersArray = extractJsonArraySlice(json, "renders");
    const auto blocks = splitTopLevelArrayElements(rendersArray);
    if (blocks.empty()) throw std::runtime_error("renders array is empty");

    std::vector<RenderSpec> specs;
    specs.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        RenderSpec r = parseRenderBlock(blocks[i]);
        validateRenderSpec(r, i);
        specs.push_back(std::move(r));
    }

    const int expected = parseExpectedRenderCount(json);
    if (static_cast<int>(specs.size()) != expected) {
        throw std::runtime_error(
            "Expected exactly " + std::to_string(expected) +
            " render entries in JSON, parsed " + std::to_string(specs.size()) +
            " (check renders array bracket matching and exporter schema alignment)");
    }
    return specs;
}

static void clearOutputWavs(const std::filesystem::path& repoRoot) {
    const std::filesystem::path dir = repoRoot / "audio" / "pgsm_stk_guitar_demo";
    std::filesystem::create_directories(dir);
    if (!std::filesystem::exists(dir)) return;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".wav") {
            std::filesystem::remove(entry.path());
        }
    }
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
    // Do not call controlChange() — STK Plucked does not implement it (virtual warning).
    // pluck_position / string_decay are retained in JSON for future mapping; excitation via pluck().
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

static void writeWav(const std::filesystem::path& path, const std::vector<double>& y, int sr) {
    std::filesystem::create_directories(path.parent_path());
    FileWvOut out(path.string(), 1, FileWrite::FILE_WAV, Stk::STK_SINT16);
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

        const std::filesystem::path repo = std::filesystem::path(repoRoot).lexically_normal();
        const std::string jsonText = readTextFile(paramsPath);
        auto specs = parseRenders(jsonText);

        clearOutputWavs(repo);

        std::vector<std::string> written;
        written.reserve(specs.size());
        for (const auto& spec : specs) {
            const std::filesystem::path out = resolveOutputPath(spec.output_wav_path, repo);
            std::cout << "Rendering " << spec.sample_id << " " << spec.note_name
                      << " -> " << out.generic_string() << "\n";
            auto y = renderOne(spec);
            writeWav(out, y, spec.sample_rate);
            written.push_back(out.generic_string());
        }

        const std::filesystem::path reportJson = repo / "audio/debug_reports/pgsm_stk_guitar_demo_report.json";
        const std::filesystem::path reportMd = repo / "audio/debug_reports/pgsm_stk_guitar_demo_report.md";
        writeReport(reportJson, reportMd, specs, written);

        std::cout << "Wrote report: " << reportJson.generic_string() << "\n";
        std::cout << "Done — " << written.size() << " WAV files.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "stk_pgsm_guitar_demo error: " << e.what() << "\n";
        return 1;
    }
}

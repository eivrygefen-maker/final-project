#include <Stk.h>
#include <FileWvOut.h>
#include <Plucked.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace stk;

// ------------------------------
// Small helpers
// ------------------------------
static void print_help() {
    std::cout <<
R"(Usage:

Plucked string -> body coloration (FEM) -> WAV
   ./guitar_stk --fem_json FEM/outputs/rect_plate_rosewood_result.json --note_hz 110 \
                --modes 20 --skip 0 --dur 5.0 --amp 0.35 --mix 0.7 --q 95 \
                --pluck_pos 0.18 --string_sustain 0.65 --string_detune 0.0 \
                --rad_k 0.02 \
                --wet_gain 20000 \
                --out audio/A2_body.wav

Notes:
- --note_hz sets musical pitch (Hz).
- --fem_json provides body modes ("modes_hz") and optionally "mode_weights".
- If mode_weights exists, it's used as A_n (after skip/modes selection).
- Q is the wood internal Q (your literature/measured value).
- --rad_k (default 0) adds extra loss with frequency WITHOUT changing wood Q:
      1/Q_total = 1/Q_wood + rad_k*(f/1000)
- --wet_gain multiplies the body (wet) signal BEFORE mixing (useful because wet can be very quiet).
  Typical values: 1e3 .. 5e4 depending on your setup.
- --mix is dry/wet: 0 = only body (wet), 1 = only string (dry).
- We drive the body using a force-like proxy: discrete acceleration of the string signal.
- Plucked controls (mapped via controlChange, value range ~0..127):
    --pluck_pos      (0..1)  -> controlChange(4, 127*pluck_pos)
    --string_sustain (0..1)  -> controlChange(11, 127*string_sustain)
    --string_detune  (0..1)  -> controlChange(1, 127*string_detune)

Determinism:
- STK noise often uses C rand(); you can set:
    --seed 123
  to make the result more repeatable.

)";
}

static bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (key == argv[i]) return true;
    }
    return false;
}

static double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

static std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Parse number array by key:
//   "<key>": [ ... ]
static std::vector<double> parse_number_array_from_json(const std::string& jsonText, const std::string& keyName) {
    const std::string key = "\"" + keyName + "\"";
    auto kpos = jsonText.find(key);
    if (kpos == std::string::npos) return {}; // optional

    auto lbr = jsonText.find('[', kpos);
    if (lbr == std::string::npos) throw std::runtime_error("Cannot find '[' after key: " + keyName);
    auto rbr = jsonText.find(']', lbr);
    if (rbr == std::string::npos) throw std::runtime_error("Cannot find ']' for array key: " + keyName);

    std::string inside = jsonText.substr(lbr + 1, rbr - (lbr + 1));
    std::vector<double> out;
    std::stringstream ss(inside);

    while (ss.good()) {
        while (ss.good() && !std::isdigit(ss.peek()) && ss.peek()!='-' && ss.peek()!='.') ss.get();
        if (!ss.good()) break;

        double v = 0.0;
        ss >> v;
        if (ss.fail()) break;
        out.push_back(v);
    }
    return out;
}

static std::vector<double> parse_modes_hz_from_json(const std::string& jsonText) {
    auto out = parse_number_array_from_json(jsonText, "modes_hz");
    if (out.empty()) throw std::runtime_error("JSON does not contain a non-empty \"modes_hz\" array.");
    return out;
}

// ------------------------------
// Resonator bank (body coloration)
// ------------------------------
struct Biquad {
    double b0=0, b1=0, b2=0, a1=0, a2=0;
    double x1=0, x2=0, y1=0, y2=0;

    void reset() { x1=x2=y1=y2=0; }

    inline double process(double x) {
        double y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;
        return y;
    }

    void set_bandpass(double f0, double Q, double fs) {
        f0 = std::max(1.0, std::min(f0, 0.49*fs));
        Q  = std::max(0.5, Q);

        double w0 = 2.0 * M_PI * (f0 / fs);
        double alpha = std::sin(w0) / (2.0 * Q);
        double cosw0 = std::cos(w0);

        double bb0 =  alpha;
        double bb1 =  0.0;
        double bb2 = -alpha;
        double aa0 =  1.0 + alpha;
        double aa1 = -2.0 * cosw0;
        double aa2 =  1.0 - alpha;

        b0 = bb0 / aa0;
        b1 = bb1 / aa0;
        b2 = bb2 / aa0;
        a1 = aa1 / aa0;
        a2 = aa2 / aa0;
    }
};

struct ResonatorBank {
    std::vector<double> freqs;
    std::vector<double> weights;
    std::vector<double> Q_wood;
    std::vector<double> Q_total;
    std::vector<Biquad> filters;

    double rad_k = 0.00;

    void build(const std::vector<double>& modes_hz,
               const std::vector<double>* mode_weights_full,
               int skip, int n_modes,
               bool use_q_range,
               double q_single,
               double q_min, double q_max,
               const std::string& q_mode,
               bool has_seed, unsigned int q_seed,
               double rad_k_,
               double fs) {

        rad_k = std::max(0.0, rad_k_);

        int start = std::max(0, skip);
        int available = (int)modes_hz.size() - start;
        int count = (n_modes <= 0) ? available : std::min(n_modes, available);
        if (count <= 0) throw std::runtime_error("No modes available after skip/modes selection.");

        freqs.assign(modes_hz.begin() + start, modes_hz.begin() + start + count);

        // weights
        weights.assign(count, 1.0);
        if (mode_weights_full && (int)mode_weights_full->size() >= start + count) {
            for (int i = 0; i < count; ++i) weights[i] = (*mode_weights_full)[start + i];
        } else {
            for (int i = 0; i < count; ++i) weights[i] = 1.0 / (1.0 + 0.25 * i);
        }
        double wmax = 0.0;
        for (double w : weights) wmax = std::max(wmax, std::abs(w));
        if (wmax > 0.0) for (double& w : weights) w /= wmax;

        // Q selection
        if (use_q_range) {
            if (!(q_min > 0 && q_max > 0 && q_max >= q_min)) {
                throw std::runtime_error("Invalid Q range. Require q_min>0, q_max>0, q_max>=q_min.");
            }
            if (!(q_mode == "mean" || q_mode == "random")) {
                throw std::runtime_error("Invalid --q_mode. Use: mean | random");
            }
        }

        std::mt19937 rng;
        if (use_q_range && q_mode == "random") {
            if (has_seed) rng.seed(q_seed);
            else { std::random_device rd; rng.seed(rd()); }
        }
        std::uniform_real_distribution<double> unif(q_min, q_max);

        Q_wood.assign(count, q_single);
        if (use_q_range) {
            if (q_mode == "mean") {
                double q_mean = 0.5 * (q_min + q_max);
                for (int i = 0; i < count; ++i) Q_wood[i] = q_mean;
            } else {
                for (int i = 0; i < count; ++i) Q_wood[i] = unif(rng);
            }
        }

        // apply extra loss without changing wood Q
        Q_total.assign(count, 0.0);
        for (int i = 0; i < count; ++i) {
            double f = std::max(1.0, freqs[i]);
            double Qw = std::max(0.5, Q_wood[i]);

            double invQ = (1.0 / Qw) + rad_k * (f / 1000.0);
            if (invQ <= 0.0) invQ = 1.0 / Qw;
            Q_total[i] = std::max(0.5, 1.0 / invQ);
        }

        filters.resize(count);
        for (int i = 0; i < count; ++i) {
            filters[i].reset();
            filters[i].set_bandpass(freqs[i], Q_total[i], fs);
        }
    }

    inline double process(double x) {
        double y = 0.0;
        for (size_t i = 0; i < filters.size(); ++i) {
            y += weights[i] * filters[i].process(x);
        }
        y *= (filters.empty() ? 1.0 : (1.0 / std::sqrt((double)filters.size())));
        return y;
    }

    void print_summary(size_t max_items = 8) const {
        size_t n = std::min(max_items, freqs.size());
        std::cout << "Body modes summary (first " << n << "):\n";
        for (size_t i = 0; i < n; ++i) {
            double f = freqs[i];
            double Qw = (i < Q_wood.size()) ? Q_wood[i] : 0.0;
            double Qt = (i < Q_total.size()) ? Q_total[i] : 0.0;
            double tau = (f > 1e-9) ? (Qt / (M_PI * f)) : 0.0;
            double w = (i < weights.size()) ? weights[i] : 0.0;
            std::cout << "  " << (i+1)
                      << ") f=" << f
                      << " Hz,  Q_wood=" << Qw
                      << ",  Q_total=" << Qt
                      << ",  tau≈" << tau << " s"
                      << ",  A=" << w << "\n";
        }
        if (rad_k > 0.0) {
            std::cout << "  (extra loss enabled) rad_k=" << rad_k
                      << " in 1/Q_total = 1/Q_wood + rad_k*(f/1000)\n";
        }
    }
};

// ------------------------------
// Main
// ------------------------------
int main(int argc, char** argv) {
    try {
        const unsigned int sampleRate = 44100;
        Stk::setSampleRate(sampleRate);
        Stk::setRawwavePath("/home/vboxuser/stk/rawwaves");

        if (argc == 1 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
            print_help();
            return 0;
        }

        std::string s;

        // Output
        std::string outPath = "audio/out.wav";
        if (get_arg(argc, argv, "--out", s)) outPath = s;

        // Duration / amp
        double dur = 2.5;
        if (get_arg(argc, argv, "--dur", s)) dur = std::stod(s);

        double amp = 0.5;
        if (get_arg(argc, argv, "--amp", s)) amp = std::stod(s);

        // Pitch
        double note_hz = 0.0;
        bool has_note = false;
        if (get_arg(argc, argv, "--note_hz", s)) { note_hz = std::stod(s); has_note = true; }
        if (get_arg(argc, argv, "--freq", s))    { note_hz = std::stod(s); has_note = true; }

        // FEM/body args
        std::string femJsonPath;
        bool has_fem = get_arg(argc, argv, "--fem_json", femJsonPath);

        int n_modes = 20;
        if (get_arg(argc, argv, "--modes", s)) n_modes = std::stoi(s);

        int skip = 0;
        if (get_arg(argc, argv, "--skip", s)) skip = std::stoi(s);

        double mix = 0.7; // more "string" by default
        if (get_arg(argc, argv, "--mix", s)) mix = std::stod(s);
        mix = clamp01(mix);

        // Wood Q
        double Q = 50.0;
        if (get_arg(argc, argv, "--q", s)) Q = std::stod(s);

        // Optional Q range
        bool use_q_range = false;
        double q_min = 0.0, q_max = 0.0;
        if (get_arg(argc, argv, "--q_min", s)) { q_min = std::stod(s); use_q_range = true; }
        if (get_arg(argc, argv, "--q_max", s)) { q_max = std::stod(s); use_q_range = true; }

        std::string q_mode = "mean";
        if (get_arg(argc, argv, "--q_mode", s)) q_mode = s;

        bool has_seed = false;
        unsigned int q_seed = 0;
        if (get_arg(argc, argv, "--q_seed", s)) { q_seed = (unsigned int)std::stoul(s); has_seed = true; }

        // Extra frequency-dependent loss (optional)
        double rad_k = 0.0;
        if (get_arg(argc, argv, "--rad_k", s)) rad_k = std::stod(s);
        rad_k = std::max(0.0, rad_k);

        // NEW: wet/body gain
        double wet_gain = 1.0;
        if (get_arg(argc, argv, "--wet_gain", s)) wet_gain = std::stod(s);
        wet_gain = std::max(0.0, wet_gain);

        // Plucked controls (0..1)
        double pluck_pos = 0.18;
        if (get_arg(argc, argv, "--pluck_pos", s)) pluck_pos = std::stod(s);
        pluck_pos = clamp01(pluck_pos);

        double string_sustain = 0.65;
        if (get_arg(argc, argv, "--string_sustain", s)) string_sustain = std::stod(s);
        string_sustain = clamp01(string_sustain);

        double string_detune = 0.0;
        if (get_arg(argc, argv, "--string_detune", s)) string_detune = std::stod(s);
        string_detune = clamp01(string_detune);

        // Optional seed for STK randomness
        if (get_arg(argc, argv, "--seed", s)) {
            unsigned int seed = (unsigned int)std::stoul(s);
            std::srand(seed);
        }

        if (!has_note) {
            throw std::runtime_error("Missing --note_hz (or --freq). Example: --note_hz 110");
        }

        // Ensure output directory exists
        std::filesystem::path op(outPath);
        if (op.has_parent_path()) std::filesystem::create_directories(op.parent_path());

        FileWvOut output;
        output.openFile(outPath, 1, FileWrite::FILE_WAV, Stk::STK_SINT16);

        // -------------------------
        // STRING EXCITATION: STK Plucked
        // -------------------------
        Plucked plk;
        plk.setFrequency(note_hz);

        /*Control changes (per STK docs / CC mapping, value ~0..127)
        plk.controlChange(4,  127.0f * (StkFloat)pluck_pos);       // pluck/pickup position
        plk.controlChange(11, 127.0f * (StkFloat)string_sustain);  // sustain
        plk.controlChange(1,  127.0f * (StkFloat)string_detune);   // detune/stretch proxy
        */
       
        // Trigger
        plk.noteOn((StkFloat)note_hz, (StkFloat)amp);

        // -------------------------
        // BODY (FEM modes)
        // -------------------------
        ResonatorBank body;
        std::vector<double> modes;
        std::vector<double> mode_weights;
        bool has_mode_weights = false;

        if (has_fem) {
            std::string txt = read_text_file(femJsonPath);
            modes = parse_modes_hz_from_json(txt);
            mode_weights = parse_number_array_from_json(txt, "mode_weights");
            has_mode_weights = !mode_weights.empty();

            body.build(
                modes,
                has_mode_weights ? &mode_weights : nullptr,
                skip,
                n_modes,
                use_q_range,
                Q,
                q_min, q_max,
                q_mode,
                has_seed, q_seed,
                rad_k,
                (double)sampleRate
            );

            body.print_summary(8);
        }

        const unsigned long totalSamples =
            static_cast<unsigned long>(std::max(0.01, dur) * (double)sampleRate);

        // Force-like coupling: discrete acceleration of the dry signal
        double dry_prev1 = 0.0;
        double dry_prev2 = 0.0;

        for (unsigned long i = 0; i < totalSamples; ++i) {
            double dry = (double)plk.tick();

            double acc = dry - 2.0 * dry_prev1 + dry_prev2;
            dry_prev2 = dry_prev1;
            dry_prev1 = dry;

            double wet = has_fem ? body.process(acc) : 0.0;
            wet *= wet_gain; // NEW: amplify wet branch

            double y = has_fem ? (mix * dry + (1.0 - mix) * wet) : dry;

            // Soft clip
            if (y > 1.0) y = 1.0;
            if (y < -1.0) y = -1.0;

            output.tick((StkFloat)y);
        }

        std::cout << "Wrote file: " << outPath << "\n";
        std::cout << "String(plucked): note_hz=" << note_hz
                  << " amp=" << amp
                  << " pluck_pos=" << pluck_pos
                  << " sustain=" << string_sustain
                  << " detune=" << string_detune
                  << "\n";

        if (has_fem) {
            std::cout << "Body: fem_json=" << femJsonPath
                      << " modes=" << n_modes << " skip=" << skip
                      << " mix=" << mix
                      << " wet_gain=" << wet_gain
                      << " Q_wood=" << Q;
            if (rad_k > 0.0) std::cout << " rad_k=" << rad_k;
            if (has_mode_weights) std::cout << " weights=mode_weights";
            else std::cout << " weights=fallback_rolloff";
            std::cout << "\n";
        }

        return 0;
    }
    catch (StkError& e) {
        std::cerr << "STK error: " << e.getMessage() << "\n";
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Run with --help for usage.\n";
        return 1;
    }
}

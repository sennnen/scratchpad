// Fast CPU trainer for the research "student" net: (768->H)x2, SCReLU, NB output buckets.
// Sparse-input forward/backward with per-thread gradient buffers; Adam; one-cycle cosine LR.
// Records: 40-byte format from research.h. Output: STUDNT01 file (+ JSON metrics on stdout).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <sys/resource.h>

struct Rec {
    uint8_t board[32];
    uint8_t stm, result, src, rule50;
    int16_t teacher, score;
};
static_assert(sizeof(Rec) == 40);

static std::vector<Rec> load(const std::string& path, int src, double frac = 1.0, double skip = 0.0) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f) / sizeof(Rec);
    size_t first = size_t(n * skip);
    n = std::min(n - first, size_t(n * frac));
    fseek(f, long(first * sizeof(Rec)), SEEK_SET);
    std::vector<Rec> v(n);
    n = fread(v.data(), sizeof(Rec), n, f);
    v.resize(n);
    fclose(f);
    if (src >= 0) v.erase(std::remove_if(v.begin(), v.end(), [&](const Rec& r) { return r.src != src; }), v.end());
    return v;
}

constexpr int NB = 8, MAXH = 2048;
constexpr float SCALE = 340.0f;
// Feature sets: 0 = 768 piece-square (perspective relative)
//               1 = 8 king buckets x 768 with horizontal mirroring (own king on files e-h),
//                   trained with an extra shared 768-row factoriser that is folded in on export.
static int FSET = 0;
static int KB_OF[32] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
static inline int n_buckets() { return FSET == 1 ? 8 : 1; }
static inline int NF_train() { return FSET == 1 ? 768 * 9 : 768; }

// features for both perspectives; returns count (<= 64)
static inline int feats(const Rec& r, int* fs, int* fn, int& bucket) {
    int codes[64], kings[2] = {0, 0}, np = 0;
    for (int sq = 0; sq < 64; ++sq) {
        codes[sq] = (r.board[sq >> 1] >> ((sq & 1) * 4)) & 15;
        if (codes[sq] == 6) kings[0] = sq;
        if (codes[sq] == 14) kings[1] = sq;
    }
    int stm = r.stm, n = 0;
    int persp[2] = {stm, 1 - stm};
    int kb[2] = {0, 0}, mir[2] = {0, 0};
    if (FSET == 1)
        for (int p = 0; p < 2; ++p) {
            int c = persp[p], ko = kings[c] ^ (56 * c);
            mir[p] = (ko & 7) >= 4 ? 7 : 0;
            kb[p] = KB_OF[(ko >> 3) * 4 + ((ko & 7) ^ mir[p])];
        }
    for (int sq = 0; sq < 64; ++sq) {
        int code = codes[sq];
        if (!code) continue;
        ++np;
        int pt = (code & 7) - 1, col = code >> 3;
        int* out[2] = {fs, fn};
        for (int p = 0; p < 2; ++p) {
            int c = persp[p], rel = col != c;
            int base = ((rel * 6 + pt) << 6) + ((sq ^ (56 * c)) ^ mir[p]);
            out[p][n] = kb[p] * 768 + base;
            if (FSET == 1) out[p][n + 1] = 768 * 8 + base;
        }
        n += FSET == 1 ? 2 : 1;
    }
    bucket = std::clamp((np - 2) / 4, 0, NB - 1);
    return n;
}

static inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Net {
    int H;
    std::vector<float> ftW, ftB, outW, outB;
    explicit Net(int h) : H(h), ftW(size_t(NF_train()) * h), ftB(h), outW(size_t(NB) * 2 * h), outB(NB) {}
    size_t size() const { return ftW.size() + ftB.size() + outW.size() + outB.size(); }
    float* seg(int i) { return i == 0 ? ftW.data() : i == 1 ? ftB.data() : i == 2 ? outW.data() : outB.data(); }
    size_t segn(int i) const { return i == 0 ? ftW.size() : i == 1 ? ftB.size() : i == 2 ? outW.size() : outB.size(); }

    // forward for one record; fills acc (2*H) if non-null; returns output
    float forward(const Rec& r, float* acc, int* fs, int* fn, int& n, int& b) const {
        n = feats(r, fs, fn, b);
        float* a0 = acc; float* a1 = acc + H;
        std::memcpy(a0, ftB.data(), H * 4);
        std::memcpy(a1, ftB.data(), H * 4);
        for (int k = 0; k < n; ++k) {
            const float* w0 = &ftW[size_t(fs[k]) * H];
            const float* w1 = &ftW[size_t(fn[k]) * H];
            for (int i = 0; i < H; ++i) { a0[i] += w0[i]; a1[i] += w1[i]; }
        }
        const float* ow = &outW[size_t(b) * 2 * H];
        float o = outB[b];
        for (int i = 0; i < 2 * H; ++i) { float x = std::clamp(acc[i], 0.0f, 1.0f); o += x * x * ow[i]; }
        return o;
    }
    void save(const std::string& p) const {
        FILE* f = fopen(p.c_str(), "wb");
        int32_t h = H, nb = NB, fset = FSET; float sc = SCALE;
        fwrite(FSET ? "STUDNT02" : "STUDNT01", 1, 8, f);
        fwrite(&h, 4, 1, f); fwrite(&nb, 4, 1, f); fwrite(&sc, 4, 1, f);
        if (FSET) fwrite(&fset, 4, 1, f);
        if (FSET == 1) {
            std::vector<float> w(size_t(768) * 8 * H);
            for (int kb = 0; kb < 8; ++kb)
                for (int fi = 0; fi < 768; ++fi)
                    for (int i = 0; i < H; ++i)
                        w[(size_t(kb) * 768 + fi) * H + i] = ftW[(size_t(kb) * 768 + fi) * H + i] + ftW[(size_t(8 * 768) + fi) * H + i];
            fwrite(w.data(), 4, w.size(), f);
        } else fwrite(ftW.data(), 4, ftW.size(), f);
        fwrite(ftB.data(), 4, ftB.size(), f);
        fwrite(outW.data(), 4, outW.size(), f); fwrite(outB.data(), 4, outB.size(), f);
        fclose(f);
    }
};

enum Label { TEACHER, SCORE, BLEND };

static float target(const Rec& r, Label lab, float lam) {
    float t = sigm(r.teacher / SCALE);
    if (lab == TEACHER) return t;
    float s = sigm(r.score / SCALE);
    if (lab == SCORE) return s;
    return lam * s + (1 - lam) * (r.result / 2.0f);
}

struct Metrics { double mse = 0, mae = 0; size_t n = 0; };

static Metrics validate(const Net& net, const std::vector<Rec>& v, int threads) {
    std::vector<Metrics> part(threads);
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&, t] {
            std::vector<float> acc(2 * net.H);
            int fs[64], fn[64], n, b;
            for (size_t i = t; i < v.size(); i += threads) {
                float o = net.forward(v[i], acc.data(), fs, fn, n, b);
                float d = sigm(o) - sigm(v[i].teacher / SCALE);
                part[t].mse += d * d;
                part[t].mae += std::min(2000.0f, std::fabs(o * SCALE - v[i].teacher));
                part[t].n++;
            }
        });
    for (auto& x : th) x.join();
    Metrics m;
    for (auto& p : part) { m.mse += p.mse; m.mae += p.mae; m.n += p.n; }
    if (m.n) { m.mse /= m.n; m.mae /= m.n; }
    return m;
}

int main(int argc, char** argv) {
    std::vector<std::string> trainFiles;
    std::vector<std::pair<std::string, std::pair<std::string, int>>> vals;
    std::string out = "net.stud", label = "teacher";
    int src = -1, H = 256, threads = 4, batch = 16384;
    double positions = 1e7, lr = 2e-3, lam = 0.7, maxN = 0, wd = 0, frac = 1.0, skip = 0.0;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--train") trainFiles.push_back(nx());
        else if (a == "--val") { std::string s = nx(); auto e = s.find('='); auto c = s.rfind(':');
            int vs = -1; std::string p = s.substr(e + 1);
            if (c != std::string::npos && c > e) { vs = std::stoi(s.substr(c + 1)); p = s.substr(e + 1, c - e - 1); }
            vals.push_back({s.substr(0, e), {p, vs}}); }
        else if (a == "--out") out = nx();
        else if (a == "--label") label = nx();
        else if (a == "--src") src = std::stoi(nx());
        else if (a == "--H") H = std::stoi(nx());
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--batch") batch = std::stoi(nx());
        else if (a == "--positions") positions = std::stod(nx());
        else if (a == "--lr") lr = std::stod(nx());
        else if (a == "--lam") lam = std::stod(nx());
        else if (a == "--n") maxN = std::stod(nx());
        else if (a == "--wd") wd = std::stod(nx());
        else if (a == "--frac") frac = std::stod(nx());
        else if (a == "--skip") skip = std::stod(nx());
        else if (a == "--fset") FSET = std::stoi(nx());
        else if (a == "--seed") seed = std::stoull(nx());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    Label lab = label == "teacher" ? TEACHER : label == "score" ? SCORE : BLEND;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Rec> data;
    {   // reserve the upper bound first: geometric growth would transiently need ~2x the data size
        size_t total = 0;
        for (auto& f : trainFiles) {
            FILE* fp = fopen(f.c_str(), "rb");
            if (fp) { fseek(fp, 0, SEEK_END); total += size_t(ftell(fp) / sizeof(Rec) * frac) + 1; fclose(fp); }
        }
        data.reserve(total);
    }
    for (auto& f : trainFiles) { auto v = load(f, src, frac, skip); data.insert(data.end(), v.begin(), v.end()); }
    std::mt19937_64 rng(seed * 7919 + 17);
    if (maxN > 0 && maxN < data.size()) {
        std::shuffle(data.begin(), data.end(), rng);
        data.resize(size_t(maxN));
        data.shrink_to_fit();
    }
    // drop extreme labels (mates) from training on teacher/score; keep for blend via clamp in sigmoid
    size_t N = data.size();
    if (N == 0) { fprintf(stderr, "no data\n"); return 1; }

    Net net(H);
    {
        std::normal_distribution<float> nd(0, 0.05f), no(0, 1.0f / std::sqrt(2.0f * H));
        for (auto& w : net.ftW) w = nd(rng);
        if (FSET == 1)  // bucket rows start at zero: the net starts as the shared 768-feature net
            std::fill(net.ftW.begin(), net.ftW.begin() + size_t(8 * 768) * H, 0.0f);
        for (auto& w : net.outW) w = no(rng);
    }
    size_t P = net.size();
    std::vector<float> m1(P), m2(P);
    std::vector<std::vector<float>> grads(threads, std::vector<float>(P));
    std::vector<double> lossPart(threads);

    size_t steps = size_t(positions / batch);
    std::vector<uint32_t> perm(N);
    for (size_t i = 0; i < N; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    size_t ptr = 0;
    double b1 = 0.9, b2 = 0.999, eps = 1e-8;
    std::vector<std::pair<size_t, double>> hist;
    for (size_t step = 0; step < steps; ++step) {
        if (ptr + batch > N) { std::shuffle(perm.begin(), perm.end(), rng); ptr = 0; }
        const uint32_t* idx = &perm[ptr];
        ptr += batch;
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&, t] {
                auto& g = grads[t];
                std::fill(g.begin(), g.end(), 0.0f);
                float* gW = g.data(); float* gB = gW + net.ftW.size();
                float* gO = gB + H; float* gOB = gO + net.outW.size();
                std::vector<float> acc(2 * H), dacc(2 * H);
                int fs[64], fn[64], n, b;
                double ls = 0;
                for (int k = t; k < batch; k += threads) {
                    const Rec& r = data[idx[k]];
                    float o = net.forward(r, acc.data(), fs, fn, n, b);
                    float s = sigm(o), y = target(r, lab, lam);
                    float d = s - y;
                    ls += d * d;
                    float go = 2.0f * d * s * (1 - s) / batch;
                    gOB[b] += go;
                    float* gob = gO + size_t(b) * 2 * H;
                    const float* ow = &net.outW[size_t(b) * 2 * H];
                    for (int i = 0; i < 2 * H; ++i) {
                        float a = acc[i];
                        float x = std::clamp(a, 0.0f, 1.0f);
                        gob[i] += go * x * x;
                        dacc[i] = (a > 0 && a < 1) ? go * ow[i] * 2 * a : 0.0f;
                    }
                    for (int i = 0; i < H; ++i) gB[i] += dacc[i] + dacc[H + i];
                    for (int q = 0; q < n; ++q) {
                        float* g0 = gW + size_t(fs[q]) * H;
                        float* g1 = gW + size_t(fn[q]) * H;
                        for (int i = 0; i < H; ++i) { g0[i] += dacc[i]; g1[i] += dacc[H + i]; }
                    }
                }
                lossPart[t] = ls;
            });
        for (auto& x : th) x.join();
        // lr schedule: 5% linear warmup from lr/10, cosine to lr/1000
        double prog = double(step) / steps, cur;
        if (prog < 0.05) cur = lr * (0.1 + 0.9 * prog / 0.05);
        else cur = lr * (0.001 + 0.999 * 0.5 * (1 + std::cos(M_PI * (prog - 0.05) / 0.95)));
        double bc1 = 1 - std::pow(b1, step + 1), bc2 = 1 - std::pow(b2, step + 1);
        // parallel reduce + Adam
        std::vector<std::thread> th2;
        float* params[4] = {net.ftW.data(), net.ftB.data(), net.outW.data(), net.outB.data()};
        size_t offs[5] = {0, net.ftW.size(), net.ftW.size() + net.ftB.size(),
                          net.ftW.size() + net.ftB.size() + net.outW.size(), P};
        for (int t = 0; t < threads; ++t)
            th2.emplace_back([&, t] {
                size_t lo = P * t / threads, hi = P * (t + 1) / threads;
                for (size_t i = lo; i < hi; ++i) {
                    float g = 0;
                    for (int u = 0; u < threads; ++u) g += grads[u][i];
                    m1[i] = b1 * m1[i] + (1 - b1) * g;
                    m2[i] = b2 * m2[i] + (1 - b2) * g * g;
                    int s = i < offs[1] ? 0 : i < offs[2] ? 1 : i < offs[3] ? 2 : 3;
                    float& p = params[s][i - offs[s]];
                    p -= cur * ((m1[i] / bc1) / (std::sqrt(m2[i] / bc2) + eps) + wd * p);
                }
            });
        for (auto& x : th2) x.join();
        double l = 0;
        for (auto v : lossPart) l += v;
        l /= batch;
        if (step % std::max<size_t>(1, steps / 20) == 0 || step + 1 == steps) {
            hist.push_back({step, l});
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "step %zu/%zu loss %.6f lr %.2e %.1fs\n", step, steps, l, cur, el);
        }
    }
    net.save(out);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    double cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    FILE* jf = fopen((out + ".json").c_str(), "w");
    fprintf(jf, "{\n \"out\": \"%s\", \"label\": \"%s\", \"lam\": %g, \"src\": %d, \"H\": %d, \"batch\": %d,\n"
                " \"frac\": %g, \"skip\": %g, \"positions\": %g, \"lr\": %g, \"wd\": %g, \"seed\": %llu, \"n_train_unique\": %zu, \"steps\": %zu,\n"
                " \"train_files\": [", out.c_str(), label.c_str(), lam, src, H, batch, frac, skip, positions, lr, wd,
            (unsigned long long)seed, N, steps);
    for (size_t i = 0; i < trainFiles.size(); ++i) fprintf(jf, "%s\"%s\"", i ? ", " : "", trainFiles[i].c_str());
    fprintf(jf, "],\n \"wall_s\": %.2f, \"cpu_s\": %.2f, \"peak_rss_mb\": %.1f,\n \"loss_hist\": [", wall, cpu, ru.ru_maxrss / 1024.0);
    for (size_t i = 0; i < hist.size(); ++i) fprintf(jf, "%s[%zu, %.6f]", i ? ", " : "", hist[i].first, hist[i].second);
    fprintf(jf, "],\n \"val\": {");
    for (size_t i = 0; i < vals.size(); ++i) {
        auto v = load(vals[i].second.first, vals[i].second.second);
        if (v.size() > 400000) v.resize(400000);
        Metrics m = validate(net, v, threads);
        fprintf(jf, "%s\"%s\": {\"mse_sig\": %.7f, \"mae_iu\": %.2f, \"n\": %zu}", i ? ", " : "", vals[i].first.c_str(), m.mse, m.mae, m.n);
        fprintf(stderr, "val %s mse_sig %.7f mae_iu %.2f n %zu\n", vals[i].first.c_str(), m.mse, m.mae, m.n);
    }
    fprintf(jf, "}\n}\n");
    fclose(jf);
    fprintf(stderr, "wall %.1fs cpu %.1fs\n", wall, cpu);
}

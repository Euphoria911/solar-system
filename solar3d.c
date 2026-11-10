// solar3d.c - 3D solar system for the terminal. Software ray tracer, true-color sub-cell glyph output.
// build: cl /O2 /W4 /openmp /nologo solar3d.c      (/openmp is optional, it just uses every core)
//
// Every planet and moon is a lit, shadowed sphere with a procedural surface and bump-mapped relief (craters on the
// Moon, Mercury, Mars and Callisto, mountains on Earth, cracked Europa, ...). Atmosphere halos, cloud shadows,
// Saturn's ring shadows, planet/moon eclipses with soft penumbras, a comet on a real Kepler ellipse with ion and dust
// tails, a 3D asteroid belt, prominences on the Sun, a Milky Way. Perspective camera with an auto tour.
//
// Output: each terminal cell is split into 2x3 sub-pixels (Unicode sextants), 2x2 (quadrants) or 1x2 (half blocks) and
// fitted with the best two colors, 24-bit fg/bg. Edges get adaptive supersampling. Only changed cells are rewritten.
// Needs Windows Terminal or another console with VT, truecolor, UTF-8 and the Symbols for Legacy Computing glyphs.
//
// keys:  0 system view   1-8 planets   9 comet   t tour on/off (default on)   m cycle resolution mode
//        arrows or a/d/w/s orbit the camera   + / - zoom   [ / ] time speed   space pause
//        l labels   o orbit lines   q or esc quit
// dev:   solar3d.exe --frames N [--size CxR] [--t0 years] [--focus K] [--fly] [--mode half|quad|sext]
//        solar3d.exe --view K [--mode ...]     start live on view K with the tour off
//
// Scale notes: sizes and distances are artistic (a true-scale system is invisible). Orbit periods are real.
// Moon periods are slowed per system so they are watchable. Time runs slower while you are zoomed on a planet.
#define _CRT_SECURE_NO_WARNINGS
#include <conio.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

#define kPi  3.14159265f
#define kTau 6.28318531f
#define C3(r, g, b) v3((float)(r), (float)(g), (float)(b))

// ---------------------------------------------------------------- config
static const float kTanHalfFovY = 0.45f;
static const float kCellAspect = 2.0f;           // terminal cell height : width (Windows Terminal default font is 1:2)
static const float kOverviewDist = 84.0f;
static const float kTransSec = 2.6f;
static const float kOverviewYearsPerSec = 0.25f;
static const float kFocusRadPerSec = 0.25f;      // planet orbital rate while zoomed on it
static const float kMinPlanetPx = 4.0f;          // planets never render smaller than this radius in pixels
static const int   kMaxPixels = 150000;          // above this the mode steps down (sextant -> quadrant -> half)
enum { kTickMs = 33, kAsteroids = 1500, kStars = 4500, kTail = 2200 };

// ---------------------------------------------------------------- vec3
typedef struct { float x, y, z; } V3;
static V3 v3(float x, float y, float z) { V3 r; r.x = x; r.y = y; r.z = z; return r; }
static V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 mulf(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static float vlen(V3 a) { return sqrtf(dot(a, a)); }
static V3 norm(V3 a) { float l = vlen(a); return l > 1e-8f ? mulf(a, 1.0f / l) : a; }
static V3 neg(V3 a) { return v3(-a.x, -a.y, -a.z); }
static float Clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static float Mixf(float a, float b, float t) { return a + (b - a) * t; }
static V3 Mix3(V3 a, V3 b, float t) { return v3(Mixf(a.x, b.x, t), Mixf(a.y, b.y, t), Mixf(a.z, b.z, t)); }
static float Sm(float a, float b, float x) { float t = Clamp01((x - a) / (b - a)); return t * t * (3.0f - 2.0f * t); }

static uint32_t gSeed = 777u;
static float Rnd(void)
{
    gSeed = gSeed * 1664525u + 1013904223u;
    return (float)(gSeed >> 8) / 16777216.0f;
}

// ---------------------------------------------------------------- noise
static float Hash3(int x, int y, int z)
{
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFFu) / 16777216.0f;
}

static float Noise3(float x, float y, float z)
{
    float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = x - fx, ty = y - fy, tz = z - fz;
    tx = tx * tx * (3.0f - 2.0f * tx);
    ty = ty * ty * (3.0f - 2.0f * ty);
    tz = tz * tz * (3.0f - 2.0f * tz);
    float x00 = Mixf(Hash3(ix, iy, iz), Hash3(ix + 1, iy, iz), tx);
    float x10 = Mixf(Hash3(ix, iy + 1, iz), Hash3(ix + 1, iy + 1, iz), tx);
    float x01 = Mixf(Hash3(ix, iy, iz + 1), Hash3(ix + 1, iy, iz + 1), tx);
    float x11 = Mixf(Hash3(ix, iy + 1, iz + 1), Hash3(ix + 1, iy + 1, iz + 1), tx);
    return Mixf(Mixf(x00, x10, ty), Mixf(x01, x11, ty), tz);
}

static float Fbm(V3 p, int oct)
{
    float a = 0.5f, s = 0.0f, f = 1.0f, tot = 0.0f;
    for (int i = 0; i < oct; i++) {
        s += a * Noise3(p.x * f, p.y * f, p.z * f);
        tot += a;
        f *= 2.03f;
        a *= 0.5f;
    }
    return s / tot;
}

// bowl-and-rim craters on the unit sphere (nearest 2x2x2 cells), negative inside bowls
static float Craters(V3 p, float freq, float seed)
{
    V3 q = v3(p.x * freq + seed, p.y * freq + seed * 0.7f, p.z * freq - seed * 0.3f);
    float fx = floorf(q.x), fy = floorf(q.y), fz = floorf(q.z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    int sx = (q.x - fx) < 0.5f ? -1 : 1, sy = (q.y - fy) < 0.5f ? -1 : 1, sz = (q.z - fz) < 0.5f ? -1 : 1;
    float h = 0.0f;
    for (int k = 0; k < 8; k++) {
        int cx = ix + ((k & 1) ? sx : 0), cy = iy + ((k & 2) ? sy : 0), cz = iz + ((k & 4) ? sz : 0);
        if (Hash3(cx, cy, cz) < 0.40f) continue;
        V3 c = v3((float)cx + 0.25f + 0.5f * Hash3(cx + 13, cy, cz),
                  (float)cy + 0.25f + 0.5f * Hash3(cx, cy + 29, cz),
                  (float)cz + 0.25f + 0.5f * Hash3(cx, cy, cz + 57));
        float r = 0.14f + 0.24f * Hash3(cx + 5, cy + 5, cz + 5);
        float d = vlen(sub(q, c)) / r;
        if (d < 1.0f) h -= 1.0f - d * d;
        if (d > 0.8f && d < 1.4f) { float e = (d - 1.0f) * 5.0f; h += 0.55f * expf(-e * e); }
    }
    return h;
}

// 2D worley distance to the nearest feature point (for storm ovals and granulation)
static float Worley2(float u, float v)
{
    float fu = floorf(u), fv = floorf(v);
    int iu = (int)fu, iv = (int)fv;
    float best = 9.0f;
    for (int dv = -1; dv <= 1; dv++)
        for (int du = -1; du <= 1; du++) {
            int cu = iu + du, cv = iv + dv;
            float px = (float)cu + Hash3(cu, cv, 3), py = (float)cv + Hash3(cu, cv, 9);
            float d = sqrtf((u - px) * (u - px) + (v - py) * (v - py));
            if (d < best) best = d;
        }
    return best;
}

// ---------------------------------------------------------------- bodies
enum { B_SUN, B_MERCURY, B_VENUS, B_EARTH, B_MARS, B_JUPITER, B_SATURN, B_URANUS, B_NEPTUNE,
       B_MOON, B_IO, B_EUROPA, B_GANYMEDE, B_CALLISTO, B_TITAN, B_COMET, B_COUNT };

typedef struct {
    const char *name;
    int   parent;                 // -1: orbits the Sun
    float R;                      // world radius
    float orbitR, periodYr, phase, incl;
    float rev;                    // moons: real revolutions per year
    int   eclipticMoon;           // 1: orbit plane = ecliptic, 0: parent's equator
    float spinRate, tilt;
    float focusDist, focusEl, focusAz;
    // per frame
    V3 pos, axis;
    float Rd, boost, fill, spin, cs, ss, ct, st;
} Body;

static Body gB[B_COUNT] = {
    [B_SUN]      = { .name = "Sun", .parent = -1, .R = 3.6f },
    [B_MERCURY]  = { .name = "Mercury", .parent = -1, .R = 0.5f, .orbitR = 7.0f, .periodYr = 0.241f, .phase = 0.6f, .incl = 0.122f,
                     .spinRate = 0.12f, .tilt = 0.0f, .focusDist = 2.1f, .focusEl = 0.25f, .focusAz = 0.65f },
    [B_VENUS]    = { .name = "Venus", .parent = -1, .R = 0.9f, .orbitR = 10.0f, .periodYr = 0.615f, .phase = 2.3f, .incl = 0.059f,
                     .spinRate = -0.06f, .tilt = 3.09f, .focusDist = 3.8f, .focusEl = 0.25f, .focusAz = 0.65f },
    [B_EARTH]    = { .name = "Earth", .parent = -1, .R = 1.0f, .orbitR = 13.5f, .periodYr = 1.0f, .phase = 4.1f, .incl = 0.0f,
                     .spinRate = 0.55f, .tilt = 0.409f, .focusDist = 4.1f, .focusEl = 0.3f, .focusAz = 0.6f },
    [B_MARS]     = { .name = "Mars", .parent = -1, .R = 0.7f, .orbitR = 17.5f, .periodYr = 1.881f, .phase = 5.4f, .incl = 0.032f,
                     .spinRate = 0.52f, .tilt = 0.44f, .focusDist = 2.9f, .focusEl = 0.25f, .focusAz = 0.65f },
    [B_JUPITER]  = { .name = "Jupiter", .parent = -1, .R = 2.8f, .orbitR = 33.0f, .periodYr = 11.86f, .phase = 1.1f, .incl = 0.023f,
                     .spinRate = 1.3f, .tilt = 0.05f, .focusDist = 11.5f, .focusEl = 0.3f, .focusAz = 0.65f },
    [B_SATURN]   = { .name = "Saturn", .parent = -1, .R = 2.4f, .orbitR = 42.0f, .periodYr = 29.46f, .phase = 3.0f, .incl = 0.043f,
                     .spinRate = 1.2f, .tilt = 0.47f, .focusDist = 10.5f, .focusEl = 0.38f, .focusAz = 0.65f },
    [B_URANUS]   = { .name = "Uranus", .parent = -1, .R = 1.7f, .orbitR = 51.0f, .periodYr = 84.0f, .phase = 0.2f, .incl = 0.013f,
                     .spinRate = -0.7f, .tilt = 1.71f, .focusDist = 6.8f, .focusEl = 0.25f, .focusAz = 0.65f },
    [B_NEPTUNE]  = { .name = "Neptune", .parent = -1, .R = 1.65f, .orbitR = 59.0f, .periodYr = 164.8f, .phase = 5.9f, .incl = 0.031f,
                     .spinRate = 0.8f, .tilt = 0.49f, .focusDist = 6.6f, .focusEl = 0.25f, .focusAz = 0.65f },
    [B_MOON]     = { .name = "Moon", .parent = B_EARTH, .R = 0.32f, .orbitR = 2.7f, .phase = 0.0f, .incl = 0.09f, .rev = 13.37f, .eclipticMoon = 1 },
    [B_IO]       = { .name = "Io", .parent = B_JUPITER, .R = 0.24f, .orbitR = 4.8f, .phase = 0.0f, .rev = 206.5f },
    [B_EUROPA]   = { .name = "Europa", .parent = B_JUPITER, .R = 0.21f, .orbitR = 6.0f, .phase = 1.6f, .rev = 102.9f },
    [B_GANYMEDE] = { .name = "Ganymede", .parent = B_JUPITER, .R = 0.34f, .orbitR = 7.5f, .phase = 3.1f, .rev = 51.05f },
    [B_CALLISTO] = { .name = "Callisto", .parent = B_JUPITER, .R = 0.30f, .orbitR = 9.2f, .phase = 4.7f, .rev = 21.89f },
    [B_TITAN]    = { .name = "Titan", .parent = B_SATURN, .R = 0.45f, .orbitR = 8.4f, .phase = 0.5f, .rev = 22.9f },
    [B_COMET]    = { .name = "Comet", .parent = -1, .R = 0.25f, .focusDist = 6.0f, .focusEl = 0.3f, .focusAz = 1.6f },
};

static const char *kViewName[10] = { "System", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Comet" };

static const float kBump[B_COUNT] = {
    [B_MERCURY] = 0.055f, [B_EARTH] = 0.05f, [B_MARS] = 0.045f, [B_MOON] = 0.05f, [B_IO] = 0.04f,
    [B_EUROPA] = 0.06f, [B_GANYMEDE] = 0.045f, [B_CALLISTO] = 0.055f, [B_COMET] = 0.07f,
};
static const float kHalo[B_COUNT] = {
    [B_VENUS] = 0.8f, [B_EARTH] = 1.0f, [B_MARS] = 0.22f, [B_JUPITER] = 0.22f, [B_SATURN] = 0.25f,
    [B_URANUS] = 0.8f, [B_NEPTUNE] = 0.8f, [B_TITAN] = 1.0f,
};

static V3 HaloColor(int id)
{
    switch (id) {
    case B_VENUS: return C3(1.0, 0.80, 0.45);
    case B_EARTH: return C3(0.35, 0.60, 1.0);
    case B_MARS: return C3(0.85, 0.55, 0.40);
    case B_JUPITER: return C3(0.9, 0.8, 0.65);
    case B_SATURN: return C3(0.95, 0.85, 0.6);
    case B_URANUS: return C3(0.5, 0.85, 0.92);
    case B_NEPTUNE: return C3(0.35, 0.55, 1.0);
    default: return C3(0.95, 0.62, 0.22);
    }
}

static float MoonScale(int parent)
{
    // visual slow-down so moons are watchable at the zoomed-in time rate
    return parent == B_EARTH ? 0.225f : parent == B_JUPITER ? 0.0026f : 0.0037f;
}

// comet: Kepler ellipse in world units
static const float kCometA = 34.0f, kCometE = 0.84f, kCometPeriodYr = 24.0f, kCometM0 = 5.55f;
static const float kCometIncl = 0.55f, kCometOmega = 0.8f;

static void CometState(float t, V3 *pos, V3 *vel, float *rOut)
{
    float M = fmodf(kTau * t / kCometPeriodYr + kCometM0, kTau);
    float E = kPi;
    for (int i = 0; i < 14; i++)
        E -= (E - kCometE * sinf(E) - M) / (1.0f - kCometE * cosf(E));
    float X0 = kCometA * (cosf(E) - kCometE);
    float Y0 = kCometA * sqrtf(1.0f - kCometE * kCometE) * sinf(E);
    float cw = cosf(kCometOmega), sw = sinf(kCometOmega);
    float X = X0 * cw - Y0 * sw, Y = X0 * sw + Y0 * cw;
    *pos = v3(X, Y * sinf(kCometIncl), Y * cosf(kCometIncl));
    *rOut = sqrtf(X * X + Y * Y);
    float E2 = E + 0.01f;
    float Xb = kCometA * (cosf(E2) - kCometE), Yb = kCometA * sqrtf(1.0f - kCometE * kCometE) * sinf(E2);
    float X2 = Xb * cw - Yb * sw, Y2 = Xb * sw + Yb * cw;
    *vel = norm(v3(X2 - X, (Y2 - Y) * sinf(kCometIncl), (Y2 - Y) * cosf(kCometIncl)));
}

// ---------------------------------------------------------------- output modes and camera
enum { M_HALF, M_QUAD, M_SEXT };
static const char *kModeName[3] = { "half", "quad", "sext" };
static int   gModePref = M_SEXT, gMode = M_SEXT, gSx = 2, gSy = 3;
static float gPixAspect = 0.75f;                 // pixel width / pixel height
static int   gW, gH;                             // pixel buffer: cols * gSx by (rows - 1) * gSy

typedef struct { V3 pos, tgt, fwd, right, up; float tanHx, tanHy; } Cam;
static Cam gCam;
static float *gCol, *gDepth, *gMask;
static int8_t *gId;
static uint8_t *gFlag;

static void SetMode(int mode)
{
    gMode = mode;
    gSx = mode == M_HALF ? 1 : 2;
    gSy = mode == M_SEXT ? 3 : 2;
    gPixAspect = ((float)gSy / (float)gSx) / kCellAspect;
}

static void BuildCam(V3 pos, V3 tgt)
{
    gCam.pos = pos;
    gCam.tgt = tgt;
    gCam.fwd = norm(sub(tgt, pos));
    gCam.right = norm(cross(gCam.fwd, v3(0.0f, 1.0f, 0.0f)));
    gCam.up = cross(gCam.right, gCam.fwd);
    gCam.tanHy = kTanHalfFovY;
    gCam.tanHx = kTanHalfFovY * ((float)gW * gPixAspect) / (float)gH;
}

static int Project(V3 p, float *sx, float *sy, float *dist)
{
    V3 v = sub(p, gCam.pos);
    float z = dot(v, gCam.fwd);
    if (z < 0.2f) return 0;
    float x = dot(v, gCam.right) / z / gCam.tanHx;
    float y = dot(v, gCam.up) / z / gCam.tanHy;
    *sx = (x * 0.5f + 0.5f) * (float)gW;
    *sy = (0.5f - y * 0.5f) * (float)gH;
    *dist = vlen(v);
    return 1;
}

// ---------------------------------------------------------------- scene update
static void ToBasis(V3 n, V3 *e1, V3 *e2)
{
    *e1 = norm(cross(n, v3(0.0f, 0.0f, 1.0f)));
    *e2 = cross(n, *e1);
}

static void PrepAxes(void)
{
    for (int i = 0; i < B_COUNT; i++) {
        Body *b = &gB[i];
        b->axis = v3(-sinf(b->tilt), cosf(b->tilt), 0.0f);
        b->ct = cosf(b->tilt);
        b->st = sinf(b->tilt);
    }
}

static void UpdateBodies(float t, float rt)
{
    PrepAxes();
    for (int i = 1; i <= B_NEPTUNE; i++) {
        Body *b = &gB[i];
        float th = b->phase + kTau * t / b->periodYr;
        b->pos = v3(b->orbitR * cosf(th), b->orbitR * sinf(th) * sinf(b->incl), b->orbitR * sinf(th) * cosf(b->incl));
        b->spin = b->spinRate * rt;
    }
    gB[B_SUN].pos = v3(0.0f, 0.0f, 0.0f);
    gB[B_SUN].spin = 0.0f;
    for (int i = B_MOON; i <= B_TITAN; i++) {
        Body *b = &gB[i];
        const Body *par = &gB[b->parent];
        float phi = b->phase + kTau * b->rev * MoonScale(b->parent) * t;
        V3 n = b->eclipticMoon ? norm(v3(0.0f, cosf(b->incl), sinf(b->incl))) : par->axis;
        V3 e1, e2;
        ToBasis(n, &e1, &e2);
        b->pos = add(par->pos, add(mulf(e1, b->orbitR * cosf(phi)), mulf(e2, b->orbitR * sinf(phi))));
        b->spin = phi + kPi;                       // tidally locked
    }
    V3 cv;
    float cr;
    CometState(t, &gB[B_COMET].pos, &cv, &cr);
    gB[B_COMET].spin = rt * 0.3f;
    for (int i = 0; i < B_COUNT; i++) {
        gB[i].cs = cosf(gB[i].spin);
        gB[i].ss = sinf(gB[i].spin);
    }
}

// display radius: planets never shrink below kMinPlanetPx; distant moons are hidden
static void SetDisplayRadii(void)
{
    for (int i = 0; i < B_COUNT; i++) {
        Body *b = &gB[i];
        float dist = vlen(sub(b->pos, gCam.pos));
        float wpp = dist * 2.0f * gCam.tanHy / (float)gH;     // world units per pixel (vertical) at that distance
        if (i == B_SUN) b->Rd = b->R > 2.6f * wpp ? b->R : 2.6f * wpp;
        else if (i >= B_MOON && i <= B_TITAN) {
            float pd = vlen(sub(gB[b->parent].pos, gCam.pos));
            b->Rd = pd > 45.0f ? 0.0f : (b->R > 1.3f * wpp ? b->R : 1.3f * wpp);
        } else if (i == B_COMET) b->Rd = b->R > 1.4f * wpp ? b->R : 1.4f * wpp;
        else b->Rd = b->R > kMinPlanetPx * wpp ? b->R : kMinPlanetPx * wpp;
        b->boost = (i == B_SUN) ? 1.0f : 1.0f + 0.9f * Sm(1.0f, 1.6f, b->Rd / b->R);
        b->fill = (i == B_SUN) ? 0.0f : 1.0f - Sm(3.0f, 9.0f, b->Rd / wpp);
    }
}

// ---------------------------------------------------------------- ray helpers
static int RaySphere(V3 o, V3 d, V3 c, float r, float *t)
{
    V3 oc = sub(o, c);
    float b = dot(oc, d);
    float cc = dot(oc, oc) - r * r;
    float disc = b * b - cc;
    if (disc < 0.0f) return 0;
    float s = sqrtf(disc);
    float t0 = -b - s;
    if (t0 > 1e-4f) { *t = t0; return 1; }
    float t1 = -b + s;
    if (t1 > 1e-4f) { *t = t1; return 1; }
    return 0;
}

static void RingProps(float rho, float *alpha, V3 *col)
{
    *alpha = 0.0f;
    if (rho < 1.24f || rho > 2.27f) return;
    float a;
    V3 c;
    if (rho < 1.52f)      { a = 0.10f + 0.18f * (rho - 1.24f) / 0.28f; c = C3(0.42, 0.38, 0.33); }
    else if (rho < 1.95f) { a = 0.88f; c = C3(0.86, 0.78, 0.64); }
    else if (rho < 2.03f) { a = 0.03f; c = C3(0.5, 0.45, 0.4); }
    else                  { a = 0.58f; c = C3(0.72, 0.66, 0.56); }
    if (rho > 2.205f && rho < 2.225f) a = 0.04f;
    float n = Noise3(rho * 60.0f, 0.5f, 0.5f) * 0.6f + Noise3(rho * 170.0f, 1.5f, 0.5f) * 0.4f;
    a *= 0.62f + 0.5f * n;
    c = mulf(c, 0.80f + 0.4f * Noise3(rho * 25.0f, 3.1f, 0.5f));
    *alpha = a > 1.0f ? 1.0f : a;
    *col = c;
}

static int RingHit(V3 o, V3 d, float *tOut, float *alpha, V3 *col)
{
    const Body *s = &gB[B_SATURN];
    float den = dot(d, s->axis);
    if (fabsf(den) < 1e-5f) return 0;
    float t = dot(sub(s->pos, o), s->axis) / den;
    if (t < 1e-3f) return 0;
    V3 q = add(o, mulf(d, t));
    float rho = vlen(sub(q, s->pos)) / s->Rd;
    RingProps(rho, alpha, col);
    if (*alpha < 0.01f) return 0;
    *tOut = t;
    return 1;
}

// soft-edged sphere shadow along the ray p -> Sun (penumbra grows with distance from the occluder)
static float SphereShadow(V3 p, V3 L, V3 c, float r)
{
    V3 oc = sub(c, p);
    float tca = dot(oc, L);
    if (tca <= 0.0f) return 1.0f;
    float d2 = dot(oc, oc) - tca * tca;
    float d = sqrtf(d2 > 0.0f ? d2 : 0.0f);
    float w = 0.03f * r + tca * 0.04f;
    return 0.04f + 0.96f * Sm(r - w, r + w, d);
}

static float Shadow(V3 p, V3 L, int self)
{
    float s = 1.0f;
    for (int i = 1; i < B_COUNT; i++) {
        if (i == self || gB[i].Rd <= 0.0f) continue;
        float f = SphereShadow(p, L, gB[i].pos, gB[i].Rd);
        if (f < s) s = f;
    }
    float t, a;
    V3 c;
    if (RingHit(p, L, &t, &a, &c)) s *= 1.0f - 0.9f * a;
    return s;
}

// ---------------------------------------------------------------- surfaces
typedef struct { V3 alb; float spec; V3 emis; float atmo; V3 atmoCol; } Surf;

static V3 RotY(V3 v, float a)
{
    float c = cosf(a), s = sinf(a);
    return v3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

static V3 ToObj(const Body *b, V3 n)
{
    V3 a = v3(n.x * b->ct + n.y * b->st, -n.x * b->st + n.y * b->ct, n.z);
    return v3(a.x * b->cs - a.z * b->ss, a.y, a.x * b->ss + a.z * b->cs);
}

static float WrapAngle(float a)
{
    while (a > kPi) a -= kTau;
    while (a < -kPi) a += kTau;
    return a;
}

static float AngDist(V3 n, float lat0, float lon0)
{
    V3 v = v3(cosf(lat0) * cosf(lon0), sinf(lat0), cosf(lat0) * sinf(lon0));
    float d = dot(n, v);
    return acosf(d < -1.0f ? -1.0f : (d > 1.0f ? 1.0f : d));
}

static float EuropaCracks(V3 n)
{
    float c1 = fabsf(Noise3(n.x * 9.0f, n.y * 9.0f, n.z * 9.0f) - 0.5f);
    float c2 = fabsf(Noise3(n.x * 17.0f + 3.0f, n.y * 17.0f, n.z * 17.0f) - 0.5f);
    return 1.0f - Sm(0.0f, 0.035f, c1 < c2 ? c1 : c2);
}

// relief height for bump mapping, object-space unit normal in
static float Height(int id, V3 n)
{
    switch (id) {
    case B_MOON:
        return 0.5f * Craters(n, 5.0f, 1.0f) + 0.3f * Craters(n, 12.0f, 4.0f) + 0.18f * Craters(n, 27.0f, 9.0f) + 0.25f * Fbm(mulf(n, 6.0f), 3);
    case B_MERCURY:
        return 0.6f * Craters(n, 6.0f, 2.0f) + 0.35f * Craters(n, 14.0f, 7.0f) + 0.2f * Craters(n, 30.0f, 3.0f) + 0.2f * Fbm(mulf(n, 8.0f), 3);
    case B_MARS: {
        float a = AngDist(n, 0.35f, 3.8f);
        return 0.35f * Craters(n, 6.0f, 5.0f) + 0.2f * Craters(n, 15.0f, 8.0f)
             + 0.5f * Fbm(add(mulf(n, 3.2f), v3(11.0f, 3.0f, 5.0f)), 4) + 1.6f * expf(-(a / 0.12f) * (a / 0.12f));
    }
    case B_CALLISTO:
        return 0.6f * Craters(n, 7.0f, 6.0f) + 0.4f * Craters(n, 16.0f, 2.0f) + 0.25f * Craters(n, 34.0f, 1.0f);
    case B_GANYMEDE:
        return 0.3f * Craters(n, 6.0f, 3.0f) + 0.35f * Fbm(mulf(n, 10.0f), 3) + 0.12f * sinf(Noise3(n.x * 5.0f, n.y * 5.0f, n.z * 5.0f) * 28.0f);
    case B_IO:
        return 0.3f * Fbm(mulf(n, 5.0f), 3);
    case B_EUROPA:
        return -0.7f * EuropaCracks(n);
    case B_COMET:
        return 1.4f * Fbm(mulf(n, 5.0f), 4) + 0.4f * Craters(n, 10.0f, 3.0f);
    case B_EARTH: {
        float land = Fbm(add(mulf(n, 2.3f), v3(3.1f, 1.7f, 5.3f)), 5);
        return land > 0.5f ? 0.9f * Sm(0.5f, 0.8f, land) * Fbm(mulf(n, 9.0f), 3) : 0.0f;
    }
    default: return 0.0f;
    }
}

static V3 Bump(const Body *b, int id, V3 n)
{
    float k = kBump[id];
    if (k <= 0.0f || b->fill > 0.85f) return n;
    V3 t1 = fabsf(n.y) < 0.95f ? norm(cross(n, v3(0.0f, 1.0f, 0.0f))) : norm(cross(n, v3(1.0f, 0.0f, 0.0f)));
    V3 t2 = cross(n, t1);
    const float e = 0.012f;
    float h0 = Height(id, ToObj(b, n));
    float h1 = Height(id, ToObj(b, norm(add(n, mulf(t1, e)))));
    float h2 = Height(id, ToObj(b, norm(add(n, mulf(t2, e)))));
    V3 g = add(mulf(t1, (h1 - h0) / e), mulf(t2, (h2 - h0) / e));
    return norm(sub(n, mulf(g, k)));
}

static Surf Tex(int id, V3 n, V3 Lo, float rt)
{
    Surf s;
    memset(&s, 0, sizeof s);
    s.alb = C3(0.5, 0.5, 0.5);
    float lat = asinf(n.y < -1.0f ? -1.0f : (n.y > 1.0f ? 1.0f : n.y));
    float lon = atan2f(n.z, n.x);
    float la = fabsf(lat);
    switch (id) {
    case B_MERCURY: {
        float f = Fbm(mulf(n, 7.0f), 5);
        float g = 0.26f + 0.40f * f;
        float cr = Craters(n, 14.0f, 7.0f);
        g *= cr < -0.35f ? 0.78f : 1.0f;                       // darker crater floors
        g *= 1.0f + 0.35f * Sm(0.35f, 0.9f, cr);               // bright rims
        float ray = Sm(0.84f, 0.92f, Noise3(n.x * 18.0f, n.y * 18.0f, n.z * 18.0f));
        s.alb = C3(g * 1.03 + ray * 0.12, g * 0.97 + ray * 0.11, g * 0.92 + ray * 0.1);
        break; }
    case B_VENUS: {
        float warp = Fbm(mulf(n, 2.5f), 3);
        float band = Fbm(v3(n.x * 2.0f + warp * 1.4f, n.y * 6.0f + warp * 1.6f, n.z * 2.0f), 5);
        float fine = Fbm(v3(n.x * 9.0f + warp * 2.0f, n.y * 16.0f, n.z * 9.0f), 3);
        s.alb = Mix3(C3(0.97, 0.85, 0.56), C3(0.80, 0.55, 0.26), band * 0.8f + fine * 0.35f);
        s.atmo = 0.9f; s.atmoCol = C3(1.0, 0.82, 0.45);
        break; }
    case B_EARTH: {
        float land = Fbm(add(mulf(n, 2.3f), v3(3.1f, 1.7f, 5.3f)), 5);
        V3 col; float spec;
        float cityMask = 0.0f;
        if (land > 0.5f) {
            float h = Clamp01((land - 0.5f) * 5.0f);
            float dry = expf(-((la - 0.5f) * (la - 0.5f)) / 0.05f) * Sm(0.40f, 0.62f, Fbm(mulf(n, 5.0f), 3));
            col = Mix3(C3(0.08, 0.32, 0.09), C3(0.22, 0.46, 0.14), Noise3(n.x * 9.0f, n.y * 9.0f, n.z * 9.0f));
            col = Mix3(col, C3(0.62, 0.50, 0.28), dry);
            col = Mix3(col, C3(0.38, 0.34, 0.30), Sm(0.55f, 1.0f, h) * 0.75f);
            col = mulf(col, 0.85f + 0.3f * Fbm(mulf(n, 22.0f), 3));
            spec = 0.03f;
            cityMask = 1.0f;
        } else {
            float d = Clamp01((0.5f - land) * 4.0f);
            col = Mix3(C3(0.08, 0.34, 0.66), C3(0.01, 0.06, 0.28), d);
            col = Mix3(col, C3(0.10, 0.45, 0.62), Sm(0.46f, 0.5f, land) * (1.0f - d) * 0.8f);   // shallow shelf
            spec = 0.9f;
        }
        float ice = Sm(1.26f, 1.36f, la + 0.14f * (Fbm(mulf(n, 6.0f), 3) - 0.5f));
        col = Mix3(col, C3(0.92, 0.94, 0.97), ice);
        spec = Mixf(spec, 0.25f, ice);
        cityMask *= 1.0f - ice;
        V3 cn = RotY(n, rt * 0.03f);
        float cl = Sm(0.50f, 0.72f, Fbm(add(mulf(cn, 3.4f), v3(7.0f, 1.0f, 3.0f)), 5));
        // cloud shadow: sample the cloud field shifted away from the Sun
        V3 sh = norm(sub(n, mulf(sub(Lo, mulf(n, dot(Lo, n))), 0.035f)));
        V3 cs = RotY(sh, rt * 0.03f);
        float clS = Sm(0.50f, 0.72f, Fbm(add(mulf(cs, 3.4f), v3(7.0f, 1.0f, 3.0f)), 4));
        col = mulf(col, 1.0f - 0.5f * clS * (1.0f - cl));
        col = Mix3(col, C3(0.96, 0.96, 0.98), cl * 0.93f);
        spec *= 1.0f - cl;
        if (cityMask > 0.5f && Noise3(n.x * 70.0f, n.y * 70.0f, n.z * 70.0f) > 0.78f && Fbm(mulf(n, 7.0f), 3) > 0.45f)
            s.emis = mulf(C3(1.0, 0.72, 0.30), 1.1f * (1.0f - cl));
        s.alb = col; s.spec = spec;
        s.atmo = 1.0f; s.atmoCol = C3(0.35, 0.60, 1.0);
        break; }
    case B_MARS: {
        float f = Fbm(add(mulf(n, 3.2f), v3(11.0f, 3.0f, 5.0f)), 5);
        V3 col = Mix3(C3(0.60, 0.27, 0.13), C3(0.80, 0.47, 0.26), f);
        float d = Fbm(add(mulf(n, 2.0f), v3(1.0f, 9.0f, 4.0f)), 4);
        col = Mix3(col, C3(0.26, 0.13, 0.09), Sm(0.52f, 0.66f, d) * 0.8f);
        float dl = WrapAngle(lon - 5.2f);                                   // Valles Marineris
        float canyon = fabsf(dl) < 0.55f ? 1.0f - Sm(0.0f, 0.04f, fabsf(lat + 0.1f + 0.04f * sinf(dl * 10.0f))) : 0.0f;
        col = Mix3(col, C3(0.22, 0.10, 0.07), canyon * 0.85f * (1.0f - Sm(0.35f, 0.55f, fabsf(dl))));
        float oly = AngDist(n, 0.35f, 3.8f);                                // Olympus Mons
        col = Mix3(col, C3(0.85, 0.62, 0.48), expf(-(oly / 0.11f) * (oly / 0.11f)) * 0.5f);
        col = Mix3(col, C3(0.93, 0.93, 0.96), Sm(1.38f, 1.46f, la + 0.08f * f));
        s.alb = col; s.atmo = 0.25f; s.atmoCol = C3(0.85, 0.55, 0.40);
        break; }
    case B_JUPITER: {
        float turb = Fbm(v3(n.x * 4.0f, n.y * 14.0f, n.z * 4.0f), 5);
        float latw = lat + 0.10f * (Fbm(v3(n.x * 5.0f, n.y * 9.0f, n.z * 5.0f), 3) - 0.5f);
        float band = sinf(latw * 10.5f + turb * 2.4f);
        V3 col = Mix3(C3(0.58, 0.36, 0.24), C3(0.95, 0.87, 0.73), 0.5f + 0.5f * band);
        float belt = Sm(0.65f, 0.95f, sinf(latw * 21.0f + turb * 3.5f));
        col = Mix3(col, C3(0.42, 0.25, 0.18), belt * 0.55f);
        float streak = Fbm(v3(lon * 2.0f, lat * 40.0f, 1.0f), 3);
        col = mulf(col, 0.88f + 0.24f * streak);
        float w = Worley2(lon * 4.0f, lat * 11.0f);                           // small white storm ovals
        col = Mix3(col, C3(0.97, 0.94, 0.88), (1.0f - Sm(0.0f, 0.16f, w)) * 0.55f * Sm(0.55f, 0.75f, turb));
        float dl = WrapAngle(lon - 1.0f);
        float e = (dl / 0.30f) * (dl / 0.30f) + ((lat + 0.40f) / 0.15f) * ((lat + 0.40f) / 0.15f);
        if (e < 1.5f) {
            float swirl = 0.5f + 0.5f * sinf(atan2f(lat + 0.40f, dl * 0.5f) * 3.0f + e * 9.0f);
            V3 spot = Mix3(C3(0.78, 0.32, 0.20), C3(0.90, 0.55, 0.38), swirl * 0.6f);
            col = Mix3(col, spot, (e < 1.0f ? sqrtf(1.0f - e) * 0.95f : 0.0f) + (e >= 1.0f ? (1.5f - e) * 0.3f : 0.0f));
        }
        s.alb = col; s.atmo = 0.25f; s.atmoCol = C3(0.9, 0.8, 0.65);
        break; }
    case B_SATURN: {
        float turb = Fbm(v3(n.x * 3.0f, n.y * 12.0f, n.z * 3.0f), 4);
        float t = 0.5f + 0.5f * sinf(lat * 12.0f + turb * 1.6f);
        V3 col = Mix3(C3(0.74, 0.63, 0.43), C3(0.93, 0.85, 0.64), t);
        col = mulf(col, 0.92f + 0.16f * Fbm(v3(lon * 2.0f, lat * 36.0f, 2.0f), 3));
        col = mulf(col, 1.0f - 0.25f * Sm(0.9f, 1.5f, la));
        if (lat > 1.35f) {                                                    // north polar hexagon
            float th = WrapAngle(lon) + kPi;
            float ph = fmodf(th, kPi / 3.0f) - kPi / 6.0f;
            float rp = kPi * 0.5f - lat;
            float hex = 0.17f / cosf(ph);
            col = Mix3(col, C3(0.42, 0.48, 0.55), (1.0f - Sm(hex - 0.015f, hex + 0.015f, rp)) * 0.75f);
        }
        s.alb = col; s.atmo = 0.3f; s.atmoCol = C3(0.95, 0.85, 0.6);
        break; }
    case B_URANUS: {
        float f = 0.92f + 0.08f * sinf(lat * 9.0f + Fbm(mulf(n, 3.0f), 3) * 2.0f);
        V3 col = mulf(C3(0.55, 0.82, 0.86), f);
        col = Mix3(col, C3(0.75, 0.92, 0.95), Sm(0.9f, 1.4f, la) * 0.5f);
        s.alb = col;
        s.atmo = 0.8f; s.atmoCol = C3(0.5, 0.85, 0.92);
        break; }
    case B_NEPTUNE: {
        float f = Fbm(v3(n.x * 2.0f, n.y * 9.0f, n.z * 2.0f), 4);
        V3 col = Mix3(C3(0.10, 0.20, 0.70), C3(0.26, 0.46, 0.96), f);
        float streak = Sm(0.70f, 0.78f, Fbm(v3(n.x * 3.0f, n.y * 18.0f + 3.0f, n.z * 3.0f), 3));
        col = Mix3(col, C3(0.85, 0.9, 1.0), streak * 0.55f);
        float dl = WrapAngle(lon - 2.2f);
        float e = (dl / 0.32f) * (dl / 0.32f) + ((lat + 0.35f) / 0.17f) * ((lat + 0.35f) / 0.17f);
        if (e < 1.0f) col = Mix3(col, C3(0.04, 0.08, 0.35), sqrtf(1.0f - e) * 0.8f);
        s.alb = col; s.atmo = 0.8f; s.atmoCol = C3(0.35, 0.55, 1.0);
        break; }
    case B_MOON: {
        float f = Fbm(mulf(n, 6.0f), 5);
        float g = 0.30f + 0.42f * f;
        float maria = Sm(0.50f, 0.62f, Fbm(add(mulf(n, 1.6f), v3(4.0f, 2.0f, 8.0f)), 3));
        g *= 1.0f - 0.40f * maria;
        float cr = Craters(n, 12.0f, 4.0f);
        g *= 1.0f + 0.25f * Sm(0.3f, 0.9f, cr) - 0.12f * Clamp01(-cr);
        // bright ray crater
        float a = AngDist(n, -0.75f, 0.5f);
        if (a < 1.2f) {
            V3 v0 = v3(cosf(-0.75f) * cosf(0.5f), sinf(-0.75f), cosf(-0.75f) * sinf(0.5f));
            V3 e1 = norm(cross(v0, v3(0.0f, 1.0f, 0.0f))), e2 = cross(v0, e1);
            float az = atan2f(dot(n, e2), dot(n, e1));
            float ray = powf(0.5f + 0.5f * sinf(az * 11.0f + 3.0f * Noise3(az * 2.0f, 0.5f, 0.5f)), 6.0f) * (1.0f - Sm(0.08f, 1.1f, a));
            g += 0.35f * ray + (a < 0.06f ? 0.25f : 0.0f);
        }
        s.alb = C3(g, g * 0.98, g * 0.95);
        break; }
    case B_IO: {
        float sp = Noise3(n.x * 9.0f, n.y * 9.0f, n.z * 9.0f);
        V3 col = Mix3(C3(0.90, 0.80, 0.30), C3(0.82, 0.55, 0.20), Fbm(mulf(n, 4.0f), 3));
        col = Mix3(col, C3(0.15, 0.10, 0.08), Sm(0.72f, 0.80f, sp));
        float ring = Sm(0.60f, 0.66f, sp) * (1.0f - Sm(0.66f, 0.72f, sp));      // red rim around vents
        col = Mix3(col, C3(0.85, 0.25, 0.12), ring * 0.7f);
        s.alb = col;
        break; }
    case B_EUROPA: {
        V3 col = Mix3(C3(0.88, 0.84, 0.76), C3(0.76, 0.70, 0.62), Fbm(mulf(n, 5.0f), 3));
        col = Mix3(col, C3(0.52, 0.32, 0.22), EuropaCracks(n) * 0.8f);
        s.alb = col;
        break; }
    case B_GANYMEDE: {
        float f = Fbm(mulf(n, 5.0f), 4);
        V3 col = Mix3(C3(0.32, 0.29, 0.26), C3(0.62, 0.58, 0.52), Sm(0.45f, 0.65f, f));
        float cr = Craters(n, 6.0f, 3.0f);
        col = mulf(col, 1.0f + 0.3f * Sm(0.3f, 0.9f, cr));
        s.alb = col;
        break; }
    case B_CALLISTO: {
        float f = Fbm(mulf(n, 7.0f), 4);
        float g = 0.18f + 0.22f * f;
        float cr = Craters(n, 16.0f, 2.0f);
        float bright = Sm(0.84f, 0.92f, Noise3(n.x * 20.0f, n.y * 20.0f, n.z * 20.0f)) + 0.35f * Sm(0.3f, 0.9f, cr);
        s.alb = C3(g + bright * 0.4f, g * 0.95f + bright * 0.4f, g * 0.9f + bright * 0.38f);
        break; }
    case B_TITAN: {
        float f = Fbm(mulf(n, 3.0f), 3);
        s.alb = Mix3(C3(0.86, 0.55, 0.20), C3(0.72, 0.42, 0.14), f);
        s.atmo = 1.0f; s.atmoCol = C3(0.95, 0.62, 0.22);
        break; }
    case B_COMET: {
        float f = Fbm(mulf(n, 5.0f), 4);
        float g = 0.08f + 0.18f * f;
        s.alb = C3(g * 1.1, g, g * 0.9);
        break; }
    default: break;
    }
    return s;
}

static float SunIntensity(float r) { return 1.0f / (1.0f + (r / 75.0f) * (r / 75.0f)); }

static V3 Shade(int id, V3 p, V3 n, V3 rd, float rt)
{
    const Body *b = &gB[id];
    V3 L = norm(neg(p));
    Surf s = Tex(id, ToObj(b, n), ToObj(b, L), rt);
    V3 nb = Bump(b, id, n);
    float ex = b->fill;                                        // 1 = only a few pixels wide, keep it visible
    float ndl = dot(n, L);
    float ndd = dot(nb, L);
    float wrap = 0.12f * s.atmo;
    float diff = Clamp01((ndd + wrap) / (1.0f + wrap));
    float vn = dot(nb, neg(rd));
    float fillL = ex * 0.55f * (vn > 0.0f ? vn : 0.0f);        // tiny bodies also get a camera-side fill light
    if (fillL > diff) diff = fillL;
    float sh = diff > 0.0f ? (ex > 0.3f ? 1.0f : Shadow(add(p, mulf(n, 1e-3f)), L, id)) : 0.0f;
    float I = SunIntensity(vlen(p));
    float lit = diff * sh * I;
    V3 c = mulf(s.alb, lit + (b->boost > 1.05f ? 0.06f : 0.010f));
    if (lit > 0.0f && s.spec > 0.0f) {
        V3 H = norm(sub(L, rd));
        float hd = dot(nb, H);
        float sp = powf(hd > 0.0f ? hd : 0.0f, 60.0f) * s.spec * sh * I;
        c = add(c, v3(sp, sp, sp * 0.95f));
    }
    c = add(c, mulf(s.emis, 1.0f - Clamp01(diff * sh * 4.0f)));
    if (s.atmo > 0.0f) {
        float vdn = -dot(n, rd);
        float rim = powf(1.0f - Clamp01(vdn), 3.0f);
        float lf = Clamp01(ndl * 1.3f + 0.2f) * (0.4f + 0.6f * sh) * I;
        c = add(c, mulf(s.atmoCol, rim * s.atmo * lf));
    }
    return mulf(c, b->boost);
}

static V3 SunColor(V3 n, V3 rd, float rt)
{
    float mu = Clamp01(-dot(n, rd));
    float limb = 0.5f + 0.5f * sqrtf(mu);
    V3 p = add(mulf(n, 8.0f), v3(rt * 0.15f, 0.0f, rt * 0.10f));
    float g = Fbm(p, 5);
    float cells = 1.0f - Worley2(n.x * 22.0f + n.z * 17.0f + rt * 0.05f, n.y * 22.0f + n.z * 9.0f);   // granulation cells
    float spots = Sm(0.60f, 0.70f, Fbm(add(mulf(n, 3.0f), v3(0.0f, rt * 0.02f, 5.0f)), 3));
    V3 c = Mix3(C3(1.0, 0.50, 0.12), C3(1.0, 0.92, 0.58), Clamp01(mu * 1.1f + (g - 0.5f) * 0.7f + (cells - 0.5f) * 0.25f));
    c = mulf(c, (0.86f + 0.30f * g + 0.12f * cells) * (1.10f + 0.40f * limb));
    return mulf(c, 1.0f - 0.45f * spots);
}

// faint Milky Way: a dusty band across the sky
static V3 Sky(V3 rd)
{
    float b = dot(rd, v3(0.2504f, 0.8813f, 0.4006f));
    float band = expf(-b * b * 14.0f);
    if (band < 0.03f) return v3(0.0f, 0.0f, 0.0f);
    float cloud = Fbm(mulf(rd, 3.5f), 4);
    float fine = Fbm(mulf(rd, 14.0f), 3);
    float dust = Sm(0.35f, 0.65f, Fbm(add(mulf(rd, 6.0f), v3(5.0f, 2.0f, 9.0f)), 3));
    float v = band * (0.35f + 1.1f * cloud) * (0.55f + 0.9f * fine) * (1.0f - 0.55f * dust) * 0.085f;
    return mulf(Mix3(C3(0.55, 0.62, 1.0), C3(1.0, 0.78, 0.55), Clamp01(cloud * 1.4f - 0.3f)), v);
}

static V3 Trace(V3 ro, V3 rd, float rt, float *depthOut, int *idOut)
{
    float tBest = 1e30f;
    int hit = -1;
    for (int i = 0; i < B_COUNT; i++) {
        if (gB[i].Rd <= 0.0f) continue;
        float t;
        if (RaySphere(ro, rd, gB[i].pos, gB[i].Rd, &t) && t < tBest) { tBest = t; hit = i; }
    }
    V3 col = v3(0.0f, 0.0f, 0.0f);
    float depth = 1e30f;
    int id = hit;
    if (hit == B_SUN) {
        V3 p = add(ro, mulf(rd, tBest));
        col = SunColor(norm(sub(p, gB[B_SUN].pos)), rd, rt);
        depth = tBest;
    } else if (hit > 0) {
        V3 p = add(ro, mulf(rd, tBest));
        col = Shade(hit, p, norm(sub(p, gB[hit].pos)), rd, rt);
        depth = tBest;
    } else {
        col = Sky(rd);
    }
    {
        float tr, a;
        V3 rc;
        if (RingHit(ro, rd, &tr, &a, &rc) && tr < tBest) {
            V3 q = add(ro, mulf(rd, tr));
            V3 L = norm(neg(q));
            float sh = Shadow(q, L, -1);
            float ndl = dot(gB[B_SATURN].axis, L);
            float lit = (0.55f + 0.60f * fabsf(ndl)) * sh * SunIntensity(vlen(q));
            if ((ndl > 0.0f) != (dot(gB[B_SATURN].axis, neg(rd)) > 0.0f)) lit *= 0.55f;
            col = Mix3(col, mulf(rc, lit + 0.01f), a);
            if (a > 0.5f) { depth = tr; id = 100; }
        }
    }
    // atmosphere halos just outside each planet's limb
    for (int i = 1; i <= B_TITAN; i++) {
        if (kHalo[i] <= 0.0f || gB[i].Rd <= 0.0f) continue;
        V3 oc = sub(gB[i].pos, ro);
        float tca = dot(oc, rd);
        if (tca <= 0.0f || (hit >= 0 && hit != i && tBest < tca)) continue;
        float R = gB[i].Rd, shell = R * 1.07f;
        float d2 = dot(oc, oc) - tca * tca;
        float d = sqrtf(d2 > 0.0f ? d2 : 0.0f);
        if (d >= shell || d < R) continue;
        float k = (shell - d) / (shell - R);
        V3 q = add(ro, mulf(rd, tca));
        float lf = Clamp01(dot(norm(sub(q, gB[i].pos)), norm(neg(q))) * 1.2f + 0.15f);
        col = add(col, mulf(HaloColor(i), k * k * kHalo[i] * lf * SunIntensity(vlen(q)) * gB[i].boost));
    }
    {   // solar glow: corona, halo and prominences, hidden by anything in front of the Sun
        V3 oc = neg(ro);
        float ocl2 = dot(oc, oc);
        float Rs = gB[B_SUN].R;
        float tca = dot(oc, rd);
        if (tca > 0.0f && ocl2 > Rs * Rs * 1.05f && !(hit > 0 && tBest < tca)) {
            float d2 = ocl2 - tca * tca;
            float tanA = sqrtf(d2 > 0.0f ? d2 : 0.0f) / tca;          // angular distance from the Sun's center
            float tanS = Rs / sqrtf(ocl2 - Rs * Rs);                   // angular radius of the Sun
            float s = tanA / tanS;
            if (s > 0.98f) {
                float q = (s - 1.0f) * 2.0f;
                float g = 0.45f / (1.0f + q * q) + 0.07f / (s * s);
                if (s < 2.0f) {
                    V3 nq = norm(add(ro, mulf(rd, tca)));
                    float fl = Sm(0.56f, 0.80f, Fbm(add(mulf(nq, 5.0f), v3(rt * 0.25f, 0.0f, rt * 0.1f)), 3));
                    g += 0.55f * fl * expf(-(s - 1.0f) * 3.2f);
                }
                col = add(col, mulf(C3(1.0, 0.70, 0.32), g));
            }
        }
    }
    *depthOut = depth;
    *idOut = id;
    return col;
}

// ---------------------------------------------------------------- overlays
static void Splat(int x, int y, float dist, V3 c, float a)
{
    if ((unsigned)x >= (unsigned)gW || (unsigned)y >= (unsigned)gH) return;
    int i = y * gW + x;
    if (dist > gDepth[i]) return;
    gCol[3 * i + 0] = gCol[3 * i + 0] * (1.0f - a) + c.x * a;
    gCol[3 * i + 1] = gCol[3 * i + 1] * (1.0f - a) + c.y * a;
    gCol[3 * i + 2] = gCol[3 * i + 2] * (1.0f - a) + c.z * a;
}

static void SplatAdd(int x, int y, float dist, V3 c, float a)
{
    if ((unsigned)x >= (unsigned)gW || (unsigned)y >= (unsigned)gH) return;
    int i = y * gW + x;
    if (dist > gDepth[i]) return;
    gCol[3 * i + 0] += c.x * a;
    gCol[3 * i + 1] += c.y * a;
    gCol[3 * i + 2] += c.z * a;
}

static void LineMask(V3 a, V3 b)
{
    float x0, y0, d0, x1, y1, d1;
    if (!Project(a, &x0, &y0, &d0) || !Project(b, &x1, &y1, &d1)) return;
    float dx = x1 - x0, dy = y1 - y0;
    float m = fabsf(dx) > fabsf(dy) ? fabsf(dx) : fabsf(dy);
    if (m > 2000.0f) return;
    int n = (int)m + 1;
    for (int i = 0; i <= n; i++) {
        float t = (float)i / (float)n;
        int x = (int)(x0 + dx * t), y = (int)(y0 + dy * t);
        if ((unsigned)x < (unsigned)gW && (unsigned)y < (unsigned)gH && d0 + (d1 - d0) * t <= gDepth[y * gW + x])
            gMask[y * gW + x] = 1.0f;
    }
}

typedef struct { V3 dir; float bright; V3 tint; } Star;
static Star gStars[kStars];
static float gAstR[kAsteroids], gAstPh[kAsteroids], gAstY[kAsteroids], gAstB[kAsteroids];
static float gTailA[kTail], gTailB[kTail], gTailC[kTail];

static void InitField(void)
{
    V3 pn = v3(0.2504f, 0.8813f, 0.4006f);
    for (int i = 0; i < kStars; i++) {
        V3 v;
        do { v = v3(Rnd() * 2.0f - 1.0f, Rnd() * 2.0f - 1.0f, Rnd() * 2.0f - 1.0f); } while (dot(v, v) > 1.0f || dot(v, v) < 0.01f);
        v = norm(v);
        if (i % 100 < 45) v = norm(sub(v, mulf(pn, dot(v, pn) * (0.80f + 0.15f * Rnd()))));
        gStars[i].dir = v;
        gStars[i].bright = 0.10f + 0.90f * powf(Rnd(), 6.0f);
        gStars[i].tint = Mix3(C3(0.75, 0.85, 1.0), C3(1.0, 0.82, 0.65), Rnd());
    }
    for (int i = 0; i < kAsteroids; i++) {
        gAstR[i] = 21.0f + 5.0f * Rnd();
        gAstPh[i] = kTau * Rnd();
        gAstY[i] = (Rnd() + Rnd() + Rnd() - 1.5f) * 0.9f;
        gAstB[i] = 0.12f + 0.22f * Rnd();
    }
    for (int i = 0; i < kTail; i++) {
        gTailA[i] = Rnd();
        gTailB[i] = Rnd() * 2.0f - 1.0f;
        gTailC[i] = Rnd() * 2.0f - 1.0f;
    }
}

static void DrawStars(float rt)
{
    for (int i = 0; i < kStars; i++) {
        float sx, sy, d;
        V3 p = add(gCam.pos, mulf(gStars[i].dir, 1000.0f));
        if (!Project(p, &sx, &sy, &d)) continue;
        float tw = 0.85f + 0.15f * sinf(rt * (1.0f + (float)(i % 5)) + (float)i);
        float b = gStars[i].bright * tw;
        int x = (int)sx, y = (int)sy;
        SplatAdd(x, y, 5000.0f, gStars[i].tint, b);
        if (gStars[i].bright > 0.85f) {
            for (int k = 1; k <= 2; k++) {                       // diffraction spikes on the bright ones
                float f = b * (k == 1 ? 0.35f : 0.15f);
                SplatAdd(x + k, y, 5000.0f, gStars[i].tint, f);
                SplatAdd(x - k, y, 5000.0f, gStars[i].tint, f);
                SplatAdd(x, y + k, 5000.0f, gStars[i].tint, f);
                SplatAdd(x, y - k, 5000.0f, gStars[i].tint, f);
            }
        }
    }
}

static void DrawOrbits(float alpha)
{
    memset(gMask, 0, sizeof(float) * (size_t)gW * (size_t)gH);
    for (int i = 1; i <= B_NEPTUNE; i++) {
        const Body *b = &gB[i];
        V3 prev = v3(0.0f, 0.0f, 0.0f);
        const int n = 540;
        for (int k = 0; k <= n; k++) {
            float th = kTau * (float)k / (float)n;
            V3 q = v3(b->orbitR * cosf(th), b->orbitR * sinf(th) * sinf(b->incl), b->orbitR * sinf(th) * cosf(b->incl));
            if (k > 0) LineMask(prev, q);
            prev = q;
        }
    }
    V3 lc = C3(0.30, 0.42, 0.75);
    for (int i = 0; i < gW * gH; i++) {
        if (gMask[i] <= 0.0f) continue;
        gCol[3 * i + 0] = gCol[3 * i + 0] * (1.0f - alpha) + lc.x * alpha;
        gCol[3 * i + 1] = gCol[3 * i + 1] * (1.0f - alpha) + lc.y * alpha;
        gCol[3 * i + 2] = gCol[3 * i + 2] * (1.0f - alpha) + lc.z * alpha;
    }
}

static void DrawAsteroids(float t)
{
    for (int i = 0; i < kAsteroids; i++) {
        float R = gAstR[i];
        float th = gAstPh[i] + kTau * t / powf(R / 13.5f, 1.5f);
        V3 p = v3(R * cosf(th), gAstY[i], R * sinf(th));
        float sx, sy, d;
        if (!Project(p, &sx, &sy, &d)) continue;
        float lit = gAstB[i] * SunIntensity(R);
        Splat((int)sx, (int)sy, d, C3(0.85 * lit, 0.78 * lit, 0.68 * lit), 0.7f);
    }
}

static void DrawComet(float t)
{
    V3 head, vel;
    float r;
    CometState(t, &head, &vel, &r);
    float intensity = Clamp01((40.0f - r) / 32.0f);
    float sx, sy, d;
    if (Project(head, &sx, &sy, &d)) {
        for (int dy = -4; dy <= 4; dy++)
            for (int dx = -4; dx <= 4; dx++) {
                float g = expf(-(float)(dx * dx + dy * dy) / 4.0f) * (0.20f + 0.60f * intensity);
                SplatAdd((int)sx + dx, (int)sy + dy, d, C3(0.6, 0.8, 1.0), g);
            }
    }
    if (intensity <= 0.01f) return;
    V3 away = norm(head);                       // from the Sun through the comet
    V3 u1 = norm(cross(away, v3(0.0f, 1.0f, 0.0f)));
    V3 u2 = cross(away, u1);
    float L = 5.0f + 30.0f * intensity;
    for (int i = 0; i < kTail; i++) {
        float s = powf(gTailA[i], 1.6f) * L;
        float spread = 0.04f + 0.07f * s;
        V3 p = add(head, mulf(away, s));
        p = add(p, add(mulf(u1, gTailB[i] * spread), mulf(u2, gTailC[i] * spread)));
        int dust = (i % 3) == 0;
        if (dust) p = sub(p, mulf(vel, s * s * 0.025f));       // dust tail curves back along the orbit
        float px, py, pd;
        if (!Project(p, &px, &py, &pd) || pd < 3.0f) continue;
        float fade = powf(1.0f - s / L, 1.3f) * intensity;
        V3 c = dust ? C3(1.0, 0.88, 0.70) : C3(0.45, 0.72, 1.0);
        SplatAdd((int)px, (int)py, pd, c, 0.30f * fade * Sm(3.0f, 14.0f, pd) * (pd > 12.0f ? 12.0f / pd : 1.0f));
    }
}

// ---------------------------------------------------------------- render passes
static V3 PixelRay(float px, float py)
{
    float u = (px / (float)gW * 2.0f - 1.0f) * gCam.tanHx;
    float v = (1.0f - py / (float)gH * 2.0f) * gCam.tanHy;
    return norm(add(gCam.fwd, add(mulf(gCam.right, u), mulf(gCam.up, v))));
}

static float Lum(const float *c) { return 0.30f * c[0] + 0.59f * c[1] + 0.11f * c[2]; }

static void RenderPixels(float rt)
{
    int y;          // OpenMP 2.0 (MSVC, C mode) wants the loop variable declared outside the for
    // pass 1: one sample per pixel
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
    for (y = 0; y < gH; y++) {
        for (int x = 0; x < gW; x++) {
            float d;
            int id;
            V3 c = Trace(gCam.pos, PixelRay((float)x + 0.5f, (float)y + 0.5f), rt, &d, &id);
            int i = y * gW + x;
            gCol[3 * i + 0] = c.x; gCol[3 * i + 1] = c.y; gCol[3 * i + 2] = c.z;
            gDepth[i] = d;
            gId[i] = (int8_t)id;
        }
    }
    // pass 2: mark edges (object changes or strong contrast)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (y = 0; y < gH; y++) {
        for (int x = 0; x < gW; x++) {
            int i = y * gW + x, f = 0;
            static const int ox[4] = { 1, -1, 0, 0 }, oy[4] = { 0, 0, 1, -1 };
            for (int k = 0; k < 4 && !f; k++) {
                int nx = x + ox[k], ny = y + oy[k];
                if (nx < 0 || ny < 0 || nx >= gW || ny >= gH) continue;
                int j = ny * gW + nx;
                if (gId[j] != gId[i] || fabsf(Lum(&gCol[3 * j]) - Lum(&gCol[3 * i])) > 0.10f) f = 1;
            }
            gFlag[i] = (uint8_t)f;
        }
    }
    // pass 3: four extra samples on the edges
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
    for (y = 0; y < gH; y++) {
        for (int x = 0; x < gW; x++) {
            int i = y * gW + x;
            if (!gFlag[i]) continue;
            static const float jx[4] = { 0.25f, 0.90f, 0.75f, 0.10f }, jy[4] = { 0.10f, 0.25f, 0.90f, 0.75f };
            V3 acc = v3(gCol[3 * i], gCol[3 * i + 1], gCol[3 * i + 2]);
            float dmin = gDepth[i];
            for (int k = 0; k < 4; k++) {
                float d;
                int id;
                V3 c = Trace(gCam.pos, PixelRay((float)x + jx[k], (float)y + jy[k]), rt, &d, &id);
                acc = add(acc, c);
                if (d < dmin) dmin = d;
            }
            acc = mulf(acc, 0.2f);
            gCol[3 * i] = acc.x; gCol[3 * i + 1] = acc.y; gCol[3 * i + 2] = acc.z;
            gDepth[i] = dmin;
        }
    }
}

// ---------------------------------------------------------------- cells
typedef struct { uint8_t fr, fg, fb, br, bg, bb; uint32_t cp; } Cell;
static Cell *gCells, *gPrev;
static uint8_t *gPix;
static int gCols, gRows;
static char *gOut;
static size_t gOutCap;

static float Tone(float c)
{
    if (c < 0.0f) c = 0.0f;
    return c < 0.8f ? c : 0.8f + 0.2f * (1.0f - expf(-(c - 0.8f) * 5.0f));
}

static void ResolvePixels(void)
{
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < gW * gH; i++) {
        for (int k = 0; k < 3; k++) {
            float c = sqrtf(Tone(gCol[3 * i + k]));
            int v = (int)(c * 255.0f + 0.5f);
            if (v > 255) v = 255;
            gPix[3 * i + k] = (uint8_t)v;
        }
    }
}

static uint32_t SextantCp(int p)    // 6-bit mask, bit0 = top-left, row-major
{
    if (p == 0 || p == 63) return ' ';
    if (p == 21) return 0x258C;
    if (p == 42) return 0x2590;
    return 0x1FB00u + (unsigned)(p - 1) - (p > 21 ? 1u : 0u) - (p > 42 ? 1u : 0u);
}

static uint32_t GlyphFor(int mode, int m)
{
    static const uint32_t quad[16] = { 0x20, 0x2598, 0x259D, 0x2580, 0x2596, 0x258C, 0x259E, 0x259B,
                                       0x2597, 0x259A, 0x2590, 0x259C, 0x2584, 0x2599, 0x259F, 0x2588 };
    if (mode == M_HALF) return m ? 0x2580 : ' ';
    if (mode == M_QUAD) return quad[m & 15];
    return SextantCp(m);
}

// best two-color fit of n sub-pixels: tries every partition with the last sub-pixel in the background group
static void FitCell(const uint8_t (*px)[3], int n, Cell *out)
{
    int best = 0, mn[3] = { 255, 255, 255 }, mx[3] = { 0, 0, 0 };
    for (int k = 0; k < n; k++)
        for (int c = 0; c < 3; c++) {
            if (px[k][c] < mn[c]) mn[c] = px[k][c];
            if (px[k][c] > mx[c]) mx[c] = px[k][c];
        }
    int sumAll[3] = { 0, 0, 0 };
    for (int k = 0; k < n; k++) for (int c = 0; c < 3; c++) sumAll[c] += px[k][c];
    int fgm[3], bgm[3];
    if (mx[0] - mn[0] < 6 && mx[1] - mn[1] < 6 && mx[2] - mn[2] < 6) {
        for (int c = 0; c < 3; c++) fgm[c] = bgm[c] = sumAll[c] / n;
    } else {
        long long bestErr = 1LL << 60;
        int nm = 1 << (n - 1);
        for (int m = 1; m < nm; m++) {
            int sa[3] = { 0, 0, 0 }, sb[3] = { 0, 0, 0 }, ca = 0, cb = 0;
            for (int k = 0; k < n; k++) {
                if ((m >> k) & 1) { for (int c = 0; c < 3; c++) sa[c] += px[k][c]; ca++; }
                else { for (int c = 0; c < 3; c++) sb[c] += px[k][c]; cb++; }
            }
            int ma[3], mb[3];
            for (int c = 0; c < 3; c++) { ma[c] = sa[c] / ca; mb[c] = sb[c] / cb; }
            long long err = 0;
            for (int k = 0; k < n; k++) {
                const int *mm = ((m >> k) & 1) ? ma : mb;
                for (int c = 0; c < 3; c++) { long long d = (long long)px[k][c] - mm[c]; err += d * d; }
            }
            if (err < bestErr) {
                bestErr = err; best = m;
                for (int c = 0; c < 3; c++) { fgm[c] = ma[c]; bgm[c] = mb[c]; }
            }
        }
        // a single flat color is cheaper to draw than a glyph
        long long flatErr = 0;
        for (int k = 0; k < n; k++)
            for (int c = 0; c < 3; c++) { long long d = (long long)px[k][c] - sumAll[c] / n; flatErr += d * d; }
        if (flatErr <= bestErr + 24) {
            best = 0;
            for (int c = 0; c < 3; c++) fgm[c] = bgm[c] = sumAll[c] / n;
        }
    }
    out->fr = (uint8_t)(fgm[0] & 0xFC); out->fg = (uint8_t)(fgm[1] & 0xFC); out->fb = (uint8_t)(fgm[2] & 0xFC);
    out->br = (uint8_t)(bgm[0] & 0xFC); out->bg = (uint8_t)(bgm[1] & 0xFC); out->bb = (uint8_t)(bgm[2] & 0xFC);
    out->cp = best ? GlyphFor(gMode, best) : ' ';
}

static void BuildCells(void)
{
    int n = gSx * gSy, cy;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (cy = 0; cy < gRows - 1; cy++) {
        for (int cx = 0; cx < gCols; cx++) {
            uint8_t px[6][3];
            for (int j = 0; j < gSy; j++)
                for (int i = 0; i < gSx; i++) {
                    const uint8_t *s = &gPix[3 * ((cy * gSy + j) * gW + cx * gSx + i)];
                    int k = j * gSx + i;
                    px[k][0] = s[0]; px[k][1] = s[1]; px[k][2] = s[2];
                }
            Cell *c = &gCells[cy * gCols + cx];
            memset(c, 0, sizeof *c);
            FitCell((const uint8_t (*)[3])px, n, c);
        }
    }
    for (int cx = 0; cx < gCols; cx++) memset(&gCells[(gRows - 1) * gCols + cx], 0, sizeof(Cell));
}

static void PutText(int cx, int cy, const char *s, uint8_t r, uint8_t g, uint8_t b)
{
    for (; *s; s++, cx++) {
        if ((unsigned)cx >= (unsigned)gCols || (unsigned)cy >= (unsigned)gRows) continue;
        Cell *c = &gCells[cy * gCols + cx];
        // dim the cell's own colors so the letter stays readable over bright backgrounds
        uint8_t nr = (uint8_t)((c->fr + c->br) / 6), ng = (uint8_t)((c->fg + c->bg) / 6), nb = (uint8_t)((c->fb + c->bb) / 6);
        c->br = nr; c->bg = ng; c->bb = nb;
        c->fr = r; c->fg = g; c->fb = b;
        c->cp = (uint32_t)(unsigned char)*s;
    }
}

static int TextFits(int cx, int cy, int len)
{
    if (cx < 0 || cx + len > gCols || cy < 0 || cy >= gRows - 1) return 0;
    for (int i = 0; i < len; i++) {
        uint32_t cp = gCells[cy * gCols + cx + i].cp;
        if (cp > ' ' && cp < 0x80) return 0;
    }
    return 1;
}

static void Label(int id)
{
    const Body *b = &gB[id];
    if (b->Rd <= 0.0f) return;
    float sx, sy, d;
    if (!Project(b->pos, &sx, &sy, &d)) return;
    int ix = (int)sx, iy = (int)sy;
    if ((unsigned)ix >= (unsigned)gW || (unsigned)iy >= (unsigned)gH) return;
    if (gDepth[iy * gW + ix] < d - b->Rd * 1.3f) return;                 // hidden behind something
    float wpp = d * 2.0f * gCam.tanHy / (float)gH;
    float rext = b->Rd * (id == B_SATURN ? 2.3f : 1.0f);
    int rcell = (int)ceilf(rext / (wpp * gPixAspect * (float)gSx));
    int len = (int)strlen(b->name);
    int cx = ix / gSx + rcell + 2, cy = iy / gSy;
    if (!TextFits(cx, cy, len)) cx = ix / gSx - rcell - 2 - len;
    if (!TextFits(cx, cy, len)) return;
    PutText(cx, cy, b->name, 205, 210, 225);
}

static void WriteUtf8(char **pp, uint32_t cp)
{
    char *p = *pp;
    if (cp < 0x80) *p++ = (char)cp;
    else if (cp < 0x800) { *p++ = (char)(0xC0 | (cp >> 6)); *p++ = (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { *p++ = (char)(0xE0 | (cp >> 12)); *p++ = (char)(0x80 | ((cp >> 6) & 63)); *p++ = (char)(0x80 | (cp & 63)); }
    else { *p++ = (char)(0xF0 | (cp >> 18)); *p++ = (char)(0x80 | ((cp >> 12) & 63)); *p++ = (char)(0x80 | ((cp >> 6) & 63)); *p++ = (char)(0x80 | (cp & 63)); }
    *pp = p;
}

static int CellEq(const Cell *a, const Cell *b)
{
    return a->fr == b->fr && a->fg == b->fg && a->fb == b->fb && a->br == b->br && a->bg == b->bg && a->bb == b->bb && a->cp == b->cp;
}

static size_t EmitFrame(int full)
{
    char *p = gOut;
    int curFg = -1, curBg = -1;
    for (int r = 0; r < gRows; r++) {
        int expect = -1;
        for (int c = 0; c < gCols; c++) {
            Cell *cell = &gCells[r * gCols + c];
            Cell *prev = &gPrev[r * gCols + c];
            if (!full && CellEq(cell, prev)) continue;
            if (expect != c) p += sprintf(p, "\x1b[%d;%dH", r + 1, c + 1);
            int bg = (cell->br << 16) | (cell->bg << 8) | cell->bb;
            int fg = (cell->fr << 16) | (cell->fg << 8) | cell->fb;
            if (bg != curBg) {
                p += sprintf(p, "\x1b[48;2;%d;%d;%dm", (bg >> 16) & 255, (bg >> 8) & 255, bg & 255);
                curBg = bg;
            }
            if (cell->cp != ' ' && fg != curFg) {
                p += sprintf(p, "\x1b[38;2;%d;%d;%dm", (fg >> 16) & 255, (fg >> 8) & 255, fg & 255);
                curFg = fg;
            }
            WriteUtf8(&p, cell->cp ? cell->cp : ' ');
            expect = c + 1;
            *prev = *cell;
        }
    }
    return (size_t)(p - gOut);
}

// ---------------------------------------------------------------- app state
static volatile LONG gRun = 1;
static BOOL WINAPI OnCtrl(DWORD type)
{
    (void)type;
    InterlockedExchange(&gRun, 0);
    return TRUE;
}

static void ConsoleSize(HANDLE hOut, int *cols, int *rows)
{
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(hOut, &csbi)) {
        *cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        *rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    } else {
        *cols = 120;
        *rows = 30;
    }
}

static int   gFocus, gTourOn = 1, gTourIdx;
static float gTourT, gSwitchT, gRealT, gSimT, gZoom = 1.0f, gAzUser, gElUser, gSpeedEff, gUserMul = 1.0f;
static V3    gStartPos, gStartTgt;
static const int   kTourFocus[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
static const float kTourDur[]   = { 14.0f, 7.0f, 7.0f, 10.0f, 7.0f, 11.0f, 11.0f, 8.0f, 8.0f, 11.0f };
#define kTourN ((int)(sizeof kTourFocus / sizeof kTourFocus[0]))

static int FocusBody(int f) { return f == 9 ? B_COMET : f; }

static float BaseSpeed(int f)
{
    if (f == 0) return kOverviewYearsPerSec;
    if (f == 9) return 0.4f;
    return kFocusRadPerSec * gB[f].periodYr / kTau;
}

static void DesiredCam(int f, V3 *pos, V3 *tgt)
{
    V3 c, dir;
    float dist;
    if (f == 0) {
        float az = 0.4f + gRealT * 0.04f + gAzUser, el = 0.55f + gElUser;
        c = v3(0.0f, 0.0f, 0.0f);
        dir = v3(cosf(el) * sinf(az), sinf(el), cosf(el) * cosf(az));
        dist = kOverviewDist * gZoom;
    } else {
        const Body *b = &gB[FocusBody(f)];
        c = b->pos;
        V3 h = norm(v3(-c.x, 0.0f, -c.z));                   // horizontal direction to the Sun
        float az = b->focusAz + gRealT * 0.02f + gAzUser, el = b->focusEl + gElUser;
        V3 d0 = RotY(h, -az);
        dir = add(mulf(d0, cosf(el)), v3(0.0f, sinf(el), 0.0f));
        dist = b->focusDist * gZoom;
    }
    *pos = add(c, mulf(norm(dir), dist));
    *tgt = c;
}

static void SwitchFocus(int f)
{
    gStartPos = gCam.pos;
    gStartTgt = gCam.tgt;
    gSwitchT = gRealT;
    gFocus = f;
    gAzUser = gElUser = 0.0f;
    gZoom = 1.0f;
}

static void UpdateCamera(void)
{
    V3 dp, dt;
    DesiredCam(gFocus, &dp, &dt);
    float b = Sm(0.0f, kTransSec, gRealT - gSwitchT);
    V3 pos = Mix3(gStartPos, dp, b);
    V3 tgt = Mix3(gStartTgt, dt, b);
    pos.y += sinf(b * kPi) * 12.0f * (gFocus == 0 ? 0.0f : 1.0f);      // arc over the plane when flying in
    for (int i = 0; i < B_COUNT; i++) {                                 // never end up inside a body
        V3 off = sub(pos, gB[i].pos);
        float l = vlen(off), minD = gB[i].R * 1.15f + 0.3f;
        if (l < minD) pos = add(gB[i].pos, mulf(l > 1e-4f ? mulf(off, 1.0f / l) : v3(0.0f, 1.0f, 0.0f), minD));
    }
    BuildCam(pos, tgt);
}

static void FreeBuffers(void)
{
    free(gCol); free(gDepth); free(gMask); free(gId); free(gFlag); free(gPix); free(gCells); free(gPrev); free(gOut);
    gCol = gDepth = gMask = NULL; gId = NULL; gFlag = NULL; gPix = NULL; gCells = gPrev = NULL; gOut = NULL;
}

static int ResizeBuffers(int cols, int rows)
{
    FreeBuffers();
    int mode = gModePref;
    for (;;) {
        SetMode(mode);
        if (mode == M_HALF || cols * gSx * (rows - 1) * gSy <= kMaxPixels) break;
        mode--;
    }
    gCols = cols;
    gRows = rows;
    gW = cols * gSx;
    gH = (rows - 1) * gSy;
    size_t np = (size_t)gW * (size_t)gH;
    gCol = malloc(sizeof(float) * 3 * np);
    gDepth = malloc(sizeof(float) * np);
    gMask = malloc(sizeof(float) * np);
    gId = malloc(np);
    gFlag = malloc(np);
    gPix = malloc(3 * np);
    gCells = calloc((size_t)cols * (size_t)rows, sizeof(Cell));
    gPrev = calloc((size_t)cols * (size_t)rows, sizeof(Cell));
    gOutCap = (size_t)cols * (size_t)rows * 64 + 4096;
    gOut = malloc(gOutCap);
    return gCol && gDepth && gMask && gId && gFlag && gPix && gCells && gPrev && gOut;
}

int main(int argc, char **argv)
{
    int testFrames = 0, forceC = 0, forceR = 0, testFocus = -1, testFly = 0, liveView = -1;
    float t0 = 0.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) testFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &forceC, &forceR);
        else if (!strcmp(argv[i], "--t0") && i + 1 < argc) t0 = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--focus") && i + 1 < argc) testFocus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--view") && i + 1 < argc) liveView = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fly")) testFly = 1;
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            ++i;
            gModePref = !strcmp(argv[i], "half") ? M_HALF : !strcmp(argv[i], "quad") ? M_QUAD : M_SEXT;
        }
    }

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  oldMode = 0;
    UINT   oldCp = GetConsoleOutputCP();
    BOOL   haveMode = GetConsoleMode(hOut, &oldMode);
    if (haveMode) {
        SetConsoleMode(hOut, oldMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        SetConsoleOutputCP(CP_UTF8);
    }
    SetConsoleCtrlHandler(OnCtrl, TRUE);
    timeBeginPeriod(1);                              // 1 ms timer so the frame pacing Sleep is accurate

    InitField();
    static const char enter[] = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J";
    DWORD wr;
    WriteFile(hOut, enter, (DWORD)(sizeof enter - 1), &wr, NULL);

    LARGE_INTEGER freq, prev, now, frameStart;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);

    int labels = 1, orbits = 1, paused = 0, frames = 0, lastC = 0, lastR = 0, forceFull = 1;
    float fps = 30.0f;
    gSimT = t0;
    gFocus = 0;
    gSpeedEff = kOverviewYearsPerSec;
    UpdateBodies(gSimT, 0.0f);
    gStartPos = v3(0.0f, 60.0f, 80.0f);
    gStartTgt = v3(0.0f, 0.0f, 0.0f);
    int startFocus = testFrames ? testFocus : liveView;
    if (startFocus >= 0 && startFocus <= 9) {
        gFocus = startFocus;
        gTourOn = 0;
        gSwitchT = -100.0f;
        gSpeedEff = BaseSpeed(gFocus);
        if (testFrames && testFly) {                // start from the system view and fly in
            DesiredCam(0, &gStartPos, &gStartTgt);
            gSwitchT = 0.0f;
        }
    }
    gCam.pos = gStartPos;
    gCam.tgt = gStartTgt;

    while (gRun) {
        QueryPerformanceCounter(&frameStart);
        if (_kbhit()) {
            int c = _getch();
            if (c == 0 || c == 224) {
                int c2 = _getch();
                if (c2 == 75) gAzUser -= 0.12f;
                else if (c2 == 77) gAzUser += 0.12f;
                else if (c2 == 72) gElUser += 0.08f;
                else if (c2 == 80) gElUser -= 0.08f;
            }
            else if (c == 'q' || c == 'Q' || c == 27) break;
            else if (c >= '0' && c <= '9') { SwitchFocus(c - '0'); gTourOn = 0; }
            else if (c == 't' || c == 'T') { gTourOn = !gTourOn; gTourIdx = 0; gTourT = 0.0f; if (gTourOn) SwitchFocus(kTourFocus[0]); }
            else if (c == 'm' || c == 'M') { gModePref = (gModePref + 2) % 3; lastC = 0; }
            else if (c == 'a' || c == 'A') gAzUser -= 0.12f;
            else if (c == 'd' || c == 'D') gAzUser += 0.12f;
            else if (c == 'w' || c == 'W') gElUser += 0.08f;
            else if (c == 's' || c == 'S') gElUser -= 0.08f;
            else if (c == '+' || c == '=') { gZoom *= 0.88f; if (gZoom < 0.2f) gZoom = 0.2f; }
            else if (c == '-' || c == '_') { gZoom *= 1.14f; if (gZoom > 4.0f) gZoom = 4.0f; }
            else if (c == ']') { gUserMul *= 1.4f; if (gUserMul > 40.0f) gUserMul = 40.0f; }
            else if (c == '[') { gUserMul /= 1.4f; if (gUserMul < 0.04f) gUserMul = 0.04f; }
            else if (c == ' ') paused = !paused;
            else if (c == 'l' || c == 'L') labels = !labels;
            else if (c == 'o' || c == 'O') orbits = !orbits;
            if (gElUser > 1.0f) gElUser = 1.0f;
            if (gElUser < -1.0f) gElUser = -1.0f;
        }

        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - prev.QuadPart) / (double)freq.QuadPart);
        prev = now;
        if (testFrames) dt = 1.0f / 30.0f;
        if (dt > 0.2f) dt = 0.2f;
        if (dt > 1e-4f) fps += (1.0f / dt - fps) * 0.1f;

        int tc, tr;
        if (forceC > 0 && forceR > 0) { tc = forceC; tr = forceR; }
        else ConsoleSize(hOut, &tc, &tr);
        if (tc < 30) tc = 30;
        if (tr < 10) tr = 10;
        if (tc != lastC || tr != lastR) {
            if (!ResizeBuffers(tc, tr)) break;
            lastC = tc;
            lastR = tr;
            forceFull = 1;
            WriteFile(hOut, "\x1b[2J", 4, &wr, NULL);
        }

        gRealT += dt;
        if (gTourOn && !testFrames) {
            gTourT += dt;
            if (gTourT > kTourDur[gTourIdx]) {
                gTourT = 0.0f;
                gTourIdx = (gTourIdx + 1) % kTourN;
                SwitchFocus(kTourFocus[gTourIdx]);
            }
        }
        float target = BaseSpeed(gFocus) * gUserMul;
        gSpeedEff += (target - gSpeedEff) * (1.0f - expf(-dt * 2.0f));
        if (!paused) gSimT += dt * gSpeedEff;

        UpdateBodies(gSimT, gRealT);
        UpdateCamera();
        SetDisplayRadii();

        RenderPixels(gRealT);
        DrawStars(gRealT);
        if (orbits) DrawOrbits(gFocus == 0 ? 0.22f : 0.09f);
        if (vlen(gCam.pos) > 30.0f) DrawAsteroids(gSimT);
        DrawComet(gSimT);
        ResolvePixels();
        BuildCells();

        if (labels) {
            if (gFocus == 0) {
                for (int i = B_EARTH; i <= B_NEPTUNE; i++) Label(i);     // Mercury and Venus sit inside the Sun's glare
                Label(B_COMET);
                Label(B_SUN);
            } else if (gFocus <= 8) {
                Label(gFocus);
                if (gFocus == 3) Label(B_MOON);
                if (gFocus == 5) for (int i = B_IO; i <= B_CALLISTO; i++) Label(i);
                if (gFocus == 6) Label(B_TITAN);
            } else {
                Label(B_COMET);
                Label(B_SUN);
            }
        }

        char hud[300];
        snprintf(hud, sizeof hud, " t=%7.2f yr  view: %-8s x%.2f%s%s  %s %dx%d %.0ffps  [0-9] view [t] tour [m] mode [arrows] cam [+/-] zoom [ [ ] ] time [space] [l] [o] [q]",
                 gSimT, kViewName[gFocus], gUserMul, paused ? " (paused)" : "", gTourOn ? " tour" : "", kModeName[gMode], gW, gH, fps);
        PutText(0, gRows - 1, hud, 140, 145, 160);

        size_t n = EmitFrame(forceFull);
        forceFull = 0;
        WriteFile(hOut, gOut, (DWORD)n, &wr, NULL);

        frames++;
        if (testFrames && frames >= testFrames) break;
        if (!testFrames) {
            QueryPerformanceCounter(&now);
            double ms = (double)(now.QuadPart - frameStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
            if (ms < kTickMs) Sleep((DWORD)(kTickMs - ms));
        }
    }

    static const char leave[] = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l";
    WriteFile(hOut, leave, (DWORD)(sizeof leave - 1), &wr, NULL);
    if (haveMode) {
        SetConsoleMode(hOut, oldMode);
        SetConsoleOutputCP(oldCp);
    }
    timeEndPeriod(1);
    FreeBuffers();
    return 0;
}

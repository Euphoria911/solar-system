// solargpu.c - 3D solar system rendered on the GPU (Direct3D 11 compute shaders). Two front ends, one ray tracer:
//
//   terminal build   cl /O2 /W4 /nologo solargpu.c
//       draws into a console with Unicode sextants (2x3 sub-pixels per cell, truecolor). A terminal is a character grid, so
//       there is always a visible pixel size: the cell grid itself.
//
//   window build     cl /O2 /W4 /nologo /DWINDOWED /Fosolarwin.obj /Fe:solarwin.exe solargpu.c /link /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup
//       draws into a native window at full resolution (no pixel blocks at all), with dynamic resolution scaling so it holds
//       60 fps, extra surface detail as you zoom in, a lens flare, mouse orbit, F11 fullscreen.
//
// Needs solargpu.hlsl next to the exe (compiled shaders are cached as .cso) and a GPU with Direct3D 11 compute (FL 11_0).
//
// The GPU ray-traces the whole scene (supersampling, procedural planets with bump-mapped relief that gets finer as you get
// closer, ray-marched atmospheric scattering, soft eclipse shadows, Saturn's ring shadows, Sun corona and prominences,
// procedural stars / Milky Way / nebulae, 3D asteroid belt and comet tail particles, bloom), tone-maps it, and in terminal
// mode fits the best two colors per cell on the GPU too. The CPU only moves the camera and presents the result.
//
// keys:  0 system view   1-8 planets   9 comet   t tour on/off (default on)   m cycle terminal mode
//        arrows or a/d/w/s orbit the camera (window: drag the mouse)   + / - zoom (window: wheel)   [ / ] time speed
//        space pause   l labels   o orbit lines   F11 fullscreen (window)   q or esc quit
// dev:   solargpu.exe --frames N [--size CxR] [--t0 years] [--focus K] [--fly] [--mode half|quad|sext]
//        solargpu.exe --view K [--fps N] [--mode ...]     start live on view K with the tour off
//        solarwin.exe --view K [--t0 years]
#define _CRT_SECURE_NO_WARNINGS
#define COBJMACROS
#include <conio.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <windowsx.h>
#include <mmsystem.h>
#pragma warning(push)
#pragma warning(disable : 4201)   // nameless unions in the Windows SDK headers
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#pragma warning(pop)
#pragma warning(disable : 4505)   // each front end leaves the other one's helpers unused
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

#define kPi  3.14159265f
#define kTau 6.28318531f

// ---------------------------------------------------------------- config
static const float kTanHalfFovY = 0.45f;
static const float kCellAspect = 2.0f;           // terminal cell height : width (Windows Terminal default font is 1:2)
static const float kOverviewDist = 84.0f;
static const float kTransSec = 2.6f;
static const float kOverviewYearsPerSec = 0.25f;
static const float kFocusRadPerSec = 0.25f;      // planet orbital rate while zoomed on it
static const float kMinPlanetPx = 4.0f;          // planets never render smaller than this radius (at 87 px screen height)
static const int   kMaxPixels = 700000;          // terminal: above this the mode steps down (sextant -> quadrant -> half)
static const float kTargetGpuMs = 11.0f;         // window: dynamic resolution aims for this GPU time per frame
enum { kAsteroids = 4500, kTail = 6000 };
static int gTickMs = 16;                          // terminal frame cap, --fps N

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
static float Clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static float Mixf(float a, float b, float t) { return a + (b - a) * t; }
static V3 Mix3(V3 a, V3 b, float t) { return v3(Mixf(a.x, b.x, t), Mixf(a.y, b.y, t), Mixf(a.z, b.z, t)); }
static float Sm(float a, float b, float x) { float t = Clamp01((x - a) / (b - a)); return t * t * (3.0f - 2.0f * t); }

static V3 RotY(V3 v, float a)
{
    float c = cosf(a), s = sinf(a);
    return v3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
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
enum { M_HALF, M_QUAD, M_SEXT, M_WIN };
static const char *kModeName[4] = { "half", "quad", "sext", "win" };
static int   gModePref = M_SEXT, gMode = M_SEXT, gSx = 2, gSy = 3, gSsaa = 3, gModeChanged;
static float gPixAspect = 0.75f;                 // pixel width / pixel height
static int   gW, gH;                             // output pixels: terminal cols * gSx by (rows - 1) * gSy, or the window client size
static int   gIW, gIH;                           // internal render resolution (window: scaled by dynamic resolution)

typedef struct { V3 pos, tgt, fwd, right, up; float tanHx, tanHy; } Cam;
static Cam gCam;

static void SetMode(int mode)
{
    gMode = mode;
    gSx = mode == M_HALF ? 1 : 2;
    gSy = mode == M_SEXT ? 3 : 2;
    if (mode == M_WIN) { gSx = gSy = 1; }
    gPixAspect = mode == M_WIN ? 1.0f : ((float)gSy / (float)gSx) / kCellAspect;
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

static void UpdateBodies(float t, float rt)
{
    for (int i = 0; i < B_COUNT; i++) {
        Body *b = &gB[i];
        b->axis = v3(-sinf(b->tilt), cosf(b->tilt), 0.0f);
        b->ct = cosf(b->tilt);
        b->st = sinf(b->tilt);
    }
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

// display radius: planets never shrink below kMinPlanetPx (scaled with screen height); distant moons are hidden
static void SetDisplayRadii(void)
{
    float sf = sqrtf((gH > 87 ? (float)gH : 87.0f) / 87.0f);
    for (int i = 0; i < B_COUNT; i++) {
        Body *b = &gB[i];
        float dist = vlen(sub(b->pos, gCam.pos));
        float wpp = dist * 2.0f * gCam.tanHy / (float)gH;     // world units per output pixel (vertical) at that distance
        if (i == B_SUN) b->Rd = b->R > 2.6f * sf * wpp ? b->R : 2.6f * sf * wpp;
        else if (i >= B_MOON && i <= B_TITAN) {
            float pd = vlen(sub(gB[b->parent].pos, gCam.pos));
            b->Rd = pd > 45.0f ? 0.0f : (b->R > 1.3f * sf * wpp ? b->R : 1.3f * sf * wpp);
        } else if (i == B_COMET) b->Rd = b->R > 1.4f * sf * wpp ? b->R : 1.4f * sf * wpp;
        else b->Rd = b->R > kMinPlanetPx * sf * wpp ? b->R : kMinPlanetPx * sf * wpp;
        b->boost = (i == B_SUN) ? 1.0f : 1.0f + 0.9f * Sm(1.0f, 1.6f, b->Rd / b->R);
        b->fill = (i == B_SUN) ? 0.0f : 1.0f - Sm(3.0f * sf, 9.0f * sf, b->Rd / wpp);
    }
}

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

// ---------------------------------------------------------------- GPU
typedef struct { float x, y, z, w; } F4;
typedef struct {
    F4 camPos, camFwd, camRight, camUp;
    int32_t dims[4], dims2[4];
    F4 bPosR[16], bAxisSpin[16], bMisc[16], bTrig[16];
    F4 orbitP[8];
    F4 cometP, cometV;
    F4 misc, misc2;
    int32_t dims3[4];
    F4 misc3;
} CB;

enum { S_RENDER, S_CLEAR, S_SCATTER, S_BRIGHT, S_BV1, S_BH2, S_BV2, S_COMPOSE, S_CELLS, S_COUNT };
static const char *kEntry[S_COUNT] = { "CS_Render", "CS_Clear", "CS_Scatter", "CS_BrightH1", "CS_BlurV1", "CS_BlurH2",
                                       "CS_BlurV2", "CS_Compose", "CS_Cells" };

#define REL(p) do { if (p) { IUnknown_Release((IUnknown *)(p)); (p) = NULL; } } while (0)

typedef ID3D11ShaderResourceView *SRVp;
typedef ID3D11UnorderedAccessView *UAVp;

static ID3D11Device *gDev;
static ID3D11DeviceContext *gCtx;
static ID3D11ComputeShader *gCS[S_COUNT];
static ID3D11Buffer *gCBuf, *gScatterBuf, *gCellsBuf, *gStage[2];
static ID3D11Texture2D *gTHdr, *gTDepth, *gTBA, *gTBB, *gTBC, *gTPix, *gTOverlay;
static SRVp gSHdr, gSDepth, gSBA, gSBB, gSBC, gSPix, gSScatter, gSOverlay;
static UAVp gUHdr, gUDepth, gUBA, gUBB, gUBC, gUPix, gUCells, gUScatter;
static char gGpuName[128] = "?";
static int  gStageValid[2], gStageIdx;
static uint32_t *gCellRaw;                    // cells * 2 uints read back from the GPU

// GPU timing (timestamp queries, read two frames later so nothing stalls)
static ID3D11Query *gQD[3], *gQ0[3], *gQ1[3];
static int gQHave[3], gQFrame;
static float gGpuMs;

static void ReleaseTargets(void)
{
    REL(gSHdr); REL(gSDepth); REL(gSBA); REL(gSBB); REL(gSBC); REL(gSPix); REL(gSScatter); REL(gSOverlay);
    REL(gUHdr); REL(gUDepth); REL(gUBA); REL(gUBB); REL(gUBC); REL(gUPix); REL(gUCells); REL(gUScatter);
    REL(gTHdr); REL(gTDepth); REL(gTBA); REL(gTBB); REL(gTBC); REL(gTPix); REL(gTOverlay);
    REL(gScatterBuf); REL(gCellsBuf); REL(gStage[0]); REL(gStage[1]);
    gStageValid[0] = gStageValid[1] = 0;
    free(gCellRaw);
    gCellRaw = NULL;
}

static ID3DBlob *LoadShader(const char *entry, const wchar_t *hlsl, const wchar_t *dir)
{
    wchar_t cso[MAX_PATH];
    swprintf(cso, MAX_PATH, L"%ls\\solargpu_%hs.cso", dir, entry);
    WIN32_FILE_ATTRIBUTE_DATA a, b;
    if (GetFileAttributesExW(cso, GetFileExInfoStandard, &a) && GetFileAttributesExW(hlsl, GetFileExInfoStandard, &b) &&
        CompareFileTime(&a.ftLastWriteTime, &b.ftLastWriteTime) > 0) {
        ID3DBlob *blob = NULL;
        if (SUCCEEDED(D3DReadFileToBlob(cso, &blob))) return blob;
    }
    ID3DBlob *code = NULL, *err = NULL;
    HRESULT hr = D3DCompileFromFile(hlsl, NULL, D3D_COMPILE_STANDARD_FILE_INCLUDE, entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (FAILED(hr)) {
        const char *msg = err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "(no message)";
        fprintf(stderr, "shader %s failed:\n%s\n", entry, msg);
        MessageBoxA(NULL, msg, "solargpu shader error", MB_OK | MB_ICONERROR);
        REL(err);
        return NULL;
    }
    REL(err);
    D3DWriteBlobToFile(code, cso, TRUE);
    return code;
}

static int InitGpu(void)
{
    IDXGIFactory1 *fac = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&fac))) { fputs("no DXGI\n", stderr); return 0; }
    IDXGIAdapter1 *best = NULL;
    SIZE_T bestMem = 0;
    DXGI_ADAPTER_DESC1 bd;
    memset(&bd, 0, sizeof bd);
    for (UINT i = 0;; i++) {
        IDXGIAdapter1 *ad = NULL;
        if (IDXGIFactory1_EnumAdapters1(fac, i, &ad) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d;
        IDXGIAdapter1_GetDesc1(ad, &d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && (d.DedicatedVideoMemory >= bestMem || !best)) {
            REL(best);
            best = ad;
            bestMem = d.DedicatedVideoMemory;
            bd = d;
        } else {
            REL(ad);
        }
    }
    REL(fac);
    if (!best) { fputs("no hardware GPU found\n", stderr); return 0; }
    WideCharToMultiByte(CP_UTF8, 0, bd.Description, -1, gGpuName, (int)sizeof gGpuName, NULL, NULL);

    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice((IDXGIAdapter *)best, D3D_DRIVER_TYPE_UNKNOWN, NULL, 0, NULL, 0, D3D11_SDK_VERSION, &gDev, &fl, &gCtx);
    REL(best);
    if (FAILED(hr)) { fprintf(stderr, "D3D11CreateDevice failed 0x%08lx\n", (unsigned long)hr); return 0; }

    wchar_t exe[MAX_PATH], dir[MAX_PATH], hlsl[MAX_PATH];
    GetModuleFileNameW(NULL, exe, MAX_PATH);
    wcscpy(dir, exe);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;
    swprintf(hlsl, MAX_PATH, L"%ls\\solargpu.hlsl", dir);
    for (int i = 0; i < S_COUNT; i++) {
        ID3DBlob *blob = LoadShader(kEntry[i], hlsl, dir);
        if (!blob) return 0;
        hr = ID3D11Device_CreateComputeShader(gDev, ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob), NULL, &gCS[i]);
        REL(blob);
        if (FAILED(hr)) { fprintf(stderr, "CreateComputeShader %s failed 0x%08lx\n", kEntry[i], (unsigned long)hr); return 0; }
    }
    D3D11_BUFFER_DESC cb;
    memset(&cb, 0, sizeof cb);
    cb.ByteWidth = (UINT)sizeof(CB);
    cb.Usage = D3D11_USAGE_DEFAULT;
    cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(ID3D11Device_CreateBuffer(gDev, &cb, NULL, &gCBuf))) return 0;

    for (int i = 0; i < 3; i++) {
        D3D11_QUERY_DESC q;
        memset(&q, 0, sizeof q);
        q.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
        ID3D11Device_CreateQuery(gDev, &q, &gQD[i]);
        q.Query = D3D11_QUERY_TIMESTAMP;
        ID3D11Device_CreateQuery(gDev, &q, &gQ0[i]);
        ID3D11Device_CreateQuery(gDev, &q, &gQ1[i]);
    }
    return 1;
}

static void GpuTimingBegin(void)
{
    int s = gQFrame % 3;
    ID3D11DeviceContext_Begin(gCtx, (ID3D11Asynchronous *)gQD[s]);
    ID3D11DeviceContext_End(gCtx, (ID3D11Asynchronous *)gQ0[s]);
}

static void GpuTimingEnd(void)
{
    int s = gQFrame % 3;
    ID3D11DeviceContext_End(gCtx, (ID3D11Asynchronous *)gQ1[s]);
    ID3D11DeviceContext_End(gCtx, (ID3D11Asynchronous *)gQD[s]);
    gQHave[s] = 1;
    int r = (gQFrame + 1) % 3;
    if (gQHave[r]) {
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
        if (ID3D11DeviceContext_GetData(gCtx, (ID3D11Asynchronous *)gQD[r], &dj, sizeof dj, D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK) {
            UINT64 a, b;
            if (ID3D11DeviceContext_GetData(gCtx, (ID3D11Asynchronous *)gQ0[r], &a, sizeof a, 0) == S_OK &&
                ID3D11DeviceContext_GetData(gCtx, (ID3D11Asynchronous *)gQ1[r], &b, sizeof b, 0) == S_OK && !dj.Disjoint && dj.Frequency) {
                float ms = (float)((double)(b - a) * 1000.0 / (double)dj.Frequency);
                gGpuMs = gGpuMs <= 0.0f ? ms : gGpuMs + (ms - gGpuMs) * 0.15f;
            }
            gQHave[r] = 0;
        }
    }
    gQFrame++;
}

static int MakeTex(DXGI_FORMAT fmt, ID3D11Texture2D **t, SRVp *s, UAVp *u)
{
    D3D11_TEXTURE2D_DESC d;
    memset(&d, 0, sizeof d);
    d.Width = (UINT)gW; d.Height = (UINT)gH; d.MipLevels = 1; d.ArraySize = 1; d.Format = fmt;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(ID3D11Device_CreateTexture2D(gDev, &d, NULL, t))) return 0;
    if (FAILED(ID3D11Device_CreateShaderResourceView(gDev, (ID3D11Resource *)*t, NULL, s))) return 0;
    if (FAILED(ID3D11Device_CreateUnorderedAccessView(gDev, (ID3D11Resource *)*t, NULL, u))) return 0;
    return 1;
}

static int MakeStructured(UINT stride, UINT count, ID3D11Buffer **b, SRVp *s, UAVp *u)
{
    D3D11_BUFFER_DESC d;
    memset(&d, 0, sizeof d);
    d.ByteWidth = stride * count;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    d.StructureByteStride = stride;
    if (FAILED(ID3D11Device_CreateBuffer(gDev, &d, NULL, b))) return 0;
    if (s) {
        D3D11_SHADER_RESOURCE_VIEW_DESC sd;
        memset(&sd, 0, sizeof sd);
        sd.Format = DXGI_FORMAT_UNKNOWN;
        sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        sd.Buffer.NumElements = count;
        if (FAILED(ID3D11Device_CreateShaderResourceView(gDev, (ID3D11Resource *)*b, &sd, s))) return 0;
    }
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud;
    memset(&ud, 0, sizeof ud);
    ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = count;
    if (FAILED(ID3D11Device_CreateUnorderedAccessView(gDev, (ID3D11Resource *)*b, &ud, u))) return 0;
    return 1;
}

// all textures are allocated at the output size; the internal (dynamic-resolution) render uses a corner of them
static int CreateTargets(int cellsX, int cellsY)
{
    ReleaseTargets();
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTHdr, &gSHdr, &gUHdr)) return 0;
    if (!MakeTex(DXGI_FORMAT_R32_FLOAT, &gTDepth, &gSDepth, &gUDepth)) return 0;
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTBA, &gSBA, &gUBA)) return 0;
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTBB, &gSBB, &gUBB)) return 0;
    if (!MakeTex(DXGI_FORMAT_R16G16B16A16_FLOAT, &gTBC, &gSBC, &gUBC)) return 0;
    if (cellsX > 0 && !MakeTex(DXGI_FORMAT_R32G32B32A32_FLOAT, &gTPix, &gSPix, &gUPix)) return 0;
    if (!MakeStructured(4, (UINT)(gW * gH * 3), &gScatterBuf, &gSScatter, &gUScatter)) return 0;
    if (cellsX > 0) {
        if (!MakeStructured(8, (UINT)(cellsX * cellsY), &gCellsBuf, NULL, &gUCells)) return 0;
        for (int i = 0; i < 2; i++) {
            D3D11_BUFFER_DESC d;
            memset(&d, 0, sizeof d);
            d.ByteWidth = (UINT)(8 * cellsX * cellsY);
            d.Usage = D3D11_USAGE_STAGING;
            d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            if (FAILED(ID3D11Device_CreateBuffer(gDev, &d, NULL, &gStage[i]))) return 0;
        }
        gCellRaw = malloc((size_t)8 * (size_t)cellsX * (size_t)cellsY);
        return gCellRaw != NULL;
    }
    return 1;
}

static void RunPass(int shader, SRVp s0, SRVp s1, SRVp s2, SRVp s3, SRVp s4, SRVp s5, SRVp s6, SRVp s7,
                    UAVp u0, UAVp u1, UAVp u2, UAVp u3, UAVp u4, UAVp u5, UINT gx, UINT gy)
{
    SRVp S[8] = { s0, s1, s2, s3, s4, s5, s6, s7 };
    UAVp U[6] = { u0, u1, u2, u3, u4, u5 };
    SRVp noS[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    UAVp noU[6] = { 0, 0, 0, 0, 0, 0 };
    ID3D11DeviceContext_CSSetShader(gCtx, gCS[shader], NULL, 0);
    ID3D11DeviceContext_CSSetShaderResources(gCtx, 0, 8, S);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(gCtx, 0, 6, U, NULL);
    ID3D11DeviceContext_Dispatch(gCtx, gx, gy, 1);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(gCtx, 0, 6, noU, NULL);
    ID3D11DeviceContext_CSSetShaderResources(gCtx, 0, 8, noS);
}

// scene, scatter and bloom at the internal resolution
static void GpuScene(const CB *cb, int nScatter)
{
    ID3D11DeviceContext_UpdateSubresource(gCtx, (ID3D11Resource *)gCBuf, 0, NULL, cb, 0, 0);
    ID3D11DeviceContext_CSSetConstantBuffers(gCtx, 0, 1, &gCBuf);
    UINT gx = (UINT)((gIW + 7) / 8), gy = (UINT)((gIH + 7) / 8);
    RunPass(S_RENDER, 0, 0, 0, 0, 0, 0, 0, 0, gUHdr, gUDepth, 0, 0, 0, 0, gx, gy);
    RunPass(S_CLEAR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gUScatter, (UINT)((gIW * gIH * 3 + 63) / 64), 1);
    if (nScatter > 0)
        RunPass(S_SCATTER, 0, 0, 0, 0, 0, gSDepth, 0, 0, 0, 0, 0, 0, 0, gUScatter, (UINT)((nScatter + 63) / 64), 1);
    RunPass(S_BRIGHT, gSHdr, 0, 0, 0, 0, 0, gSScatter, 0, 0, 0, gUBA, 0, 0, 0, gx, gy);
    RunPass(S_BV1, 0, gSBA, 0, 0, 0, 0, 0, 0, 0, 0, gUBB, 0, 0, 0, gx, gy);
    RunPass(S_BH2, 0, gSBB, 0, 0, 0, 0, 0, 0, 0, 0, gUBA, 0, 0, 0, gx, gy);
    RunPass(S_BV2, 0, gSBA, 0, 0, 0, 0, 0, 0, 0, 0, gUBC, 0, 0, 0, gx, gy);
}

// tone map into `target` (the float pixel texture for the terminal, the swap chain back buffer for the window)
static void GpuCompose(UAVp target)
{
    RunPass(S_COMPOSE, gSHdr, 0, gSBB, gSBC, 0, gSDepth, gSScatter, gSOverlay, 0, 0, 0, target, 0, 0,
            (UINT)((gW + 7) / 8), (UINT)((gH + 7) / 8));
}

// terminal: fit cells on the GPU and read them back (this frame when sync, else the previous frame)
static const uint32_t *GpuCells(int sync)
{
    RunPass(S_CELLS, 0, 0, 0, 0, gSPix, 0, 0, 0, 0, 0, 0, 0, gUCells, 0,
            (UINT)((gW / gSx + 7) / 8), (UINT)(((gH / gSy) + 7) / 8));
    int cur = gStageIdx;
    ID3D11DeviceContext_CopyResource(gCtx, (ID3D11Resource *)gStage[cur], (ID3D11Resource *)gCellsBuf);
    gStageValid[cur] = 1;
    int rd = sync ? cur : 1 - cur;
    gStageIdx = 1 - cur;
    if (!gStageValid[rd]) return NULL;
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ID3D11DeviceContext_Map(gCtx, (ID3D11Resource *)gStage[rd], 0, D3D11_MAP_READ, 0, &m))) return NULL;
    memcpy(gCellRaw, m.pData, (size_t)8 * (size_t)(gW / gSx) * (size_t)(gH / gSy));
    ID3D11DeviceContext_Unmap(gCtx, (ID3D11Resource *)gStage[rd], 0);
    return gCellRaw;
}

static F4 f4(V3 v, float w) { F4 r; r.x = v.x; r.y = v.y; r.z = v.z; r.w = w; return r; }

static void FillCB(CB *cb, float rt, float simT, int orbits, float orbitAlpha, int ssaa, int windowed, int *nScatter)
{
    memset(cb, 0, sizeof *cb);
    cb->camPos = f4(gCam.pos, rt);
    cb->camFwd = f4(gCam.fwd, simT);
    cb->camRight = f4(gCam.right, gCam.tanHx);
    cb->camUp = f4(gCam.up, gCam.tanHy);
    cb->dims[0] = gIW; cb->dims[1] = gIH; cb->dims[2] = ssaa; cb->dims[3] = gW / gSx;
    cb->dims2[0] = gH / gSy; cb->dims2[1] = gSx; cb->dims2[2] = gSy; cb->dims2[3] = orbits ? 1 : 0;
    cb->dims3[0] = gW; cb->dims3[1] = gH; cb->dims3[2] = windowed;
    for (int i = 0; i < B_COUNT; i++) {
        const Body *b = &gB[i];
        cb->bPosR[i] = f4(b->pos, b->Rd);
        cb->bAxisSpin[i] = f4(b->axis, b->spin);
        cb->bMisc[i].x = b->R; cb->bMisc[i].y = b->boost; cb->bMisc[i].z = b->fill; cb->bMisc[i].w = b->tilt;
        cb->bTrig[i].x = b->cs; cb->bTrig[i].y = b->ss; cb->bTrig[i].z = b->ct; cb->bTrig[i].w = b->st;
    }
    for (int i = 1; i <= B_NEPTUNE; i++) {
        cb->orbitP[i - 1].x = gB[i].orbitR;
        cb->orbitP[i - 1].y = gB[i].incl;
    }
    V3 head, vel;
    float r;
    CometState(simT, &head, &vel, &r);
    float intensity = Clamp01((40.0f - r) / 32.0f);
    cb->cometP = f4(head, intensity);
    cb->cometV = f4(vel, 5.0f + 30.0f * intensity);
    int nA = vlen(gCam.pos) > 30.0f ? kAsteroids : 0;
    int nT = intensity > 0.01f ? kTail : 0;
    cb->misc.x = orbitAlpha; cb->misc.y = 0.9f; cb->misc.z = 0.8f; cb->misc.w = 1.0f;
    cb->misc2.x = (float)nA; cb->misc2.y = (float)nT; cb->misc2.z = 2.0f * gCam.tanHy / (float)gIH;
    cb->misc3.x = windowed ? 0.7f : 0.0f;
    *nScatter = nA + nT;
}

// ---------------------------------------------------------------- shared input and camera
static volatile LONG gRun = 1;
static BOOL WINAPI OnCtrl(DWORD type)
{
    (void)type;
    InterlockedExchange(&gRun, 0);
    return TRUE;
}

enum { K_LEFT = 1001, K_RIGHT, K_UP, K_DOWN, K_FULL };

static int   gFocus, gTourOn = 1, gTourIdx;
static int   gLabels = 1, gOrbits = 1, gPaused = 0;
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

// advance time, the tour and the camera for one frame
static void StepWorld(float dt, int testMode)
{
    gRealT += dt;
    if (gTourOn && !testMode) {
        gTourT += dt;
        if (gTourT > kTourDur[gTourIdx]) {
            gTourT = 0.0f;
            gTourIdx = (gTourIdx + 1) % kTourN;
            SwitchFocus(kTourFocus[gTourIdx]);
        }
    }
    float target = BaseSpeed(gFocus) * gUserMul;
    gSpeedEff += (target - gSpeedEff) * (1.0f - expf(-dt * 2.0f));
    if (!gPaused) gSimT += dt * gSpeedEff;
    UpdateBodies(gSimT, gRealT);
    UpdateCamera();
    SetDisplayRadii();
}

// returns 1 to quit
static int HandleKey(int c)
{
    if (c == K_LEFT) gAzUser -= 0.12f;
    else if (c == K_RIGHT) gAzUser += 0.12f;
    else if (c == K_UP) gElUser += 0.08f;
    else if (c == K_DOWN) gElUser -= 0.08f;
    else if (c == 'q' || c == 'Q' || c == 27) return 1;
    else if (c >= '0' && c <= '9') { SwitchFocus(c - '0'); gTourOn = 0; }
    else if (c == 't' || c == 'T') { gTourOn = !gTourOn; gTourIdx = 0; gTourT = 0.0f; if (gTourOn) SwitchFocus(kTourFocus[0]); }
    else if (c == 'm' || c == 'M') { gModePref = (gModePref + 2) % 3; gModeChanged = 1; }
    else if (c == 'a' || c == 'A') gAzUser -= 0.12f;
    else if (c == 'd' || c == 'D') gAzUser += 0.12f;
    else if (c == 'w' || c == 'W') gElUser += 0.08f;
    else if (c == 's' || c == 'S') gElUser -= 0.08f;
    else if (c == '+' || c == '=') { gZoom *= 0.88f; if (gZoom < 0.2f) gZoom = 0.2f; }
    else if (c == '-' || c == '_') { gZoom *= 1.14f; if (gZoom > 4.0f) gZoom = 4.0f; }
    else if (c == ']') { gUserMul *= 1.4f; if (gUserMul > 40.0f) gUserMul = 40.0f; }
    else if (c == '[') { gUserMul /= 1.4f; if (gUserMul < 0.04f) gUserMul = 0.04f; }
    else if (c == ' ') gPaused = !gPaused;
    else if (c == 'l' || c == 'L') gLabels = !gLabels;
    else if (c == 'o' || c == 'O') gOrbits = !gOrbits;
    if (gElUser > 1.0f) gElUser = 1.0f;
    if (gElUser < -1.0f) gElUser = -1.0f;
    return 0;
}

static int Occluded(int id)
{
    V3 d = sub(gB[id].pos, gCam.pos);
    float tc = vlen(d);
    d = mulf(d, 1.0f / tc);
    for (int j = 0; j < B_COUNT; j++) {
        if (j == id || gB[j].Rd <= 0.0f) continue;
        float t;
        if (RaySphere(gCam.pos, d, gB[j].pos, gB[j].Rd, &t) && t < tc - gB[id].Rd * 1.2f) return 1;
    }
    return 0;
}

static void InitView(float t0, int startFocus, int fly)
{
    gSimT = t0;
    gFocus = 0;
    gSpeedEff = kOverviewYearsPerSec;
    UpdateBodies(gSimT, 0.0f);
    gStartPos = v3(0.0f, 60.0f, 80.0f);
    gStartTgt = v3(0.0f, 0.0f, 0.0f);
    if (startFocus >= 0 && startFocus <= 9) {
        gFocus = startFocus;
        gTourOn = 0;
        gSwitchT = -100.0f;
        gSpeedEff = BaseSpeed(gFocus);
        if (fly) {                                   // start from the system view and fly in
            DesiredCam(0, &gStartPos, &gStartTgt);
            gSwitchT = 0.0f;
        }
    }
    gCam.pos = gStartPos;
    gCam.tgt = gStartTgt;
}

// ================================================================= terminal front end
typedef struct { uint8_t fr, fg, fb, br, bg, bb; uint32_t cp; } Cell;
static Cell *gCells, *gPrev;
static int gCols, gRows;
static char *gOut;

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

static void CellsFromGpu(const uint32_t *raw)
{
    int cellsX = gCols, cellsY = gRows - 1;
    for (int i = 0; i < cellsX * cellsY; i++) {
        uint32_t w0 = raw[2 * i], w1 = raw[2 * i + 1];
        Cell *c = &gCells[i];
        memset(c, 0, sizeof *c);
        c->fr = (uint8_t)(w0 >> 16); c->fg = (uint8_t)(w0 >> 8); c->fb = (uint8_t)w0;
        c->br = (uint8_t)(w1 >> 16); c->bg = (uint8_t)(w1 >> 8); c->bb = (uint8_t)w1;
        int mask = (int)((w0 >> 24) & 63u);
        c->cp = mask ? GlyphFor(gMode, mask) : ' ';
    }
    for (int cx = 0; cx < gCols; cx++) memset(&gCells[(gRows - 1) * gCols + cx], 0, sizeof(Cell));
}

static void PutText(int cx, int cy, const char *s, uint8_t r, uint8_t g, uint8_t b)
{
    for (; *s; s++, cx++) {
        if ((unsigned)cx >= (unsigned)gCols || (unsigned)cy >= (unsigned)gRows) continue;
        Cell *c = &gCells[cy * gCols + cx];
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
    if (Occluded(id)) return;
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

static void FreeCpuBuffers(void)
{
    free(gCells); free(gPrev); free(gOut);
    gCells = gPrev = NULL; gOut = NULL;
}

static int ResizeTerminal(int cols, int rows)
{
    FreeCpuBuffers();
    int mode = gModePref;
    for (;;) {
        SetMode(mode);
        if (mode == M_HALF || cols * gSx * (rows - 1) * gSy <= kMaxPixels) break;
        mode--;
    }
    gCols = cols;
    gRows = rows;
    gW = gIW = cols * gSx;
    gH = gIH = (rows - 1) * gSy;
    gSsaa = gW * gH <= 160000 ? 3 : (gW * gH <= 400000 ? 2 : 1);
    gCells = calloc((size_t)cols * (size_t)rows, sizeof(Cell));
    gPrev = calloc((size_t)cols * (size_t)rows, sizeof(Cell));
    gOut = malloc((size_t)cols * (size_t)rows * 64 + 4096);
    return gCells && gPrev && gOut && CreateTargets(cols, rows - 1);
}

static int RunTerminal(float t0, int startFocus, int testFrames, int forceC, int forceR, int testFly)
{
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

    static const char enter[] = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J";
    DWORD wr;
    WriteFile(hOut, enter, (DWORD)(sizeof enter - 1), &wr, NULL);

    LARGE_INTEGER freq, prev, now, frameStart;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);

    int frames = 0, lastC = 0, lastR = 0, forceFull = 1, failed = 0;
    float fps = 30.0f, cpuMs = 0.0f;
    InitView(t0, startFocus, testFrames && testFly);

    while (gRun) {
        QueryPerformanceCounter(&frameStart);
        if (_kbhit()) {
            int c = _getch();
            if (c == 0 || c == 224) {
                int c2 = _getch();
                c = c2 == 75 ? K_LEFT : c2 == 77 ? K_RIGHT : c2 == 72 ? K_UP : c2 == 80 ? K_DOWN : 0;
            }
            if (c && HandleKey(c)) break;
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
        if (tc != lastC || tr != lastR || gModeChanged) {
            gModeChanged = 0;
            if (!ResizeTerminal(tc, tr)) { failed = 1; break; }
            lastC = tc;
            lastR = tr;
            forceFull = 1;
            WriteFile(hOut, "\x1b[2J", 4, &wr, NULL);
        }

        StepWorld(dt, testFrames != 0);

        CB cb;
        int nScatter = 0;
        FillCB(&cb, gRealT, gSimT, gOrbits, gFocus == 0 ? 0.22f : 0.09f, gSsaa, 0, &nScatter);
        GpuTimingBegin();
        GpuScene(&cb, nScatter);
        GpuCompose(gUPix);
        const uint32_t *raw = GpuCells(testFrames != 0);
        GpuTimingEnd();
        if (raw) {
            CellsFromGpu(raw);
            if (gLabels) {
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
            char hud[320];
            snprintf(hud, sizeof hud, " t=%7.2f yr  %-8s x%.2f%s%s  GPU %s %.1fms  %s %dx%d ss%d  %.0ffps cpu %.1fms  [0-9] view [t] tour [m] mode [arrows] cam [+/-] zoom [ [ ] ] time [space] [l] [o] [q]",
                     gSimT, kViewName[gFocus], gUserMul, gPaused ? " (paused)" : "", gTourOn ? " tour" : "", gGpuName, gGpuMs,
                     kModeName[gMode], gW, gH, gSsaa, fps, cpuMs);
            PutText(0, gRows - 1, hud, 140, 145, 160);
            size_t n = EmitFrame(forceFull);
            forceFull = 0;
            WriteFile(hOut, gOut, (DWORD)n, &wr, NULL);
        }

        frames++;
        if (testFrames && frames >= testFrames) break;
        QueryPerformanceCounter(&now);
        double ms = (double)(now.QuadPart - frameStart.QuadPart) * 1000.0 / (double)freq.QuadPart;
        cpuMs += ((float)ms - cpuMs) * 0.1f;
        if (!testFrames && ms < gTickMs) Sleep((DWORD)(gTickMs - ms));
    }

    static const char leave[] = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l";
    WriteFile(hOut, leave, (DWORD)(sizeof leave - 1), &wr, NULL);
    if (haveMode) {
        SetConsoleMode(hOut, oldMode);
        SetConsoleOutputCP(oldCp);
    }
    timeEndPeriod(1);
    ReleaseTargets();
    FreeCpuBuffers();
    if (failed) fputs("failed to allocate GPU resources\n", stderr);
    return failed;
}

// ================================================================= window front end
static HWND gHwnd;
static int  gWinW = 1600, gWinH = 900, gWinDirty, gFullscreen, gDragging, gLastMx, gLastMy;
static int  gKeyQ[128], gKeyN;
static WINDOWPLACEMENT gWp;
static IDXGISwapChain1 *gSwap;
static ID3D11Texture2D *gBackTex;
static UAVp gBackUav;

// text overlay: a GDI DIB drawn on the CPU, uploaded only where text changed, blended in the compose shader
typedef struct { int x0, y0, x1, y1; } Rect;
static HDC gDc;
static HBITMAP gDib;
static uint8_t *gDibBits;
static HFONT gFont;
static int gDibW, gDibH;
static Rect gRectPrev[48], gRectCur[48];
static int gNPrev, gNCur;

static void PushKey(int k) { if (gKeyN < 128) gKeyQ[gKeyN++] = k; }

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m) {
    case WM_SIZE:
        if (w != SIZE_MINIMIZED && LOWORD(l) > 0 && HIWORD(l) > 0) { gWinW = LOWORD(l); gWinH = HIWORD(l); gWinDirty = 1; }
        return 0;
    case WM_CLOSE:
        InterlockedExchange(&gRun, 0);
        return 0;
    case WM_CHAR:
        PushKey((int)w);
        return 0;
    case WM_KEYDOWN:
        if (w == VK_LEFT) PushKey(K_LEFT);
        else if (w == VK_RIGHT) PushKey(K_RIGHT);
        else if (w == VK_UP) PushKey(K_UP);
        else if (w == VK_DOWN) PushKey(K_DOWN);
        else if (w == VK_F11) PushKey(K_FULL);
        return 0;
    case WM_LBUTTONDOWN:
        SetCapture(h);
        gDragging = 1;
        gLastMx = GET_X_LPARAM(l);
        gLastMy = GET_Y_LPARAM(l);
        return 0;
    case WM_LBUTTONUP:
        gDragging = 0;
        ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if (gDragging) {
            int x = GET_X_LPARAM(l), y = GET_Y_LPARAM(l);
            gAzUser += (float)(x - gLastMx) * 0.006f;
            gElUser += (float)(y - gLastMy) * 0.005f;
            if (gElUser > 1.0f) gElUser = 1.0f;
            if (gElUser < -1.0f) gElUser = -1.0f;
            gLastMx = x;
            gLastMy = y;
        }
        return 0;
    case WM_MOUSEWHEEL:
        gZoom *= GET_WHEEL_DELTA_WPARAM(w) > 0 ? 0.9f : 1.11f;
        if (gZoom < 0.2f) gZoom = 0.2f;
        if (gZoom > 4.0f) gZoom = 4.0f;
        return 0;
    case WM_LBUTTONDBLCLK:
        PushKey(K_FULL);
        return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

static void ToggleFullscreen(void)
{
    DWORD style = (DWORD)GetWindowLongA(gHwnd, GWL_STYLE);
    if (!gFullscreen) {
        MONITORINFO mi;
        memset(&mi, 0, sizeof mi);
        mi.cbSize = sizeof mi;
        gWp.length = sizeof gWp;
        GetWindowPlacement(gHwnd, &gWp);
        GetMonitorInfoA(MonitorFromWindow(gHwnd, MONITOR_DEFAULTTONEAREST), &mi);
        SetWindowLongA(gHwnd, GWL_STYLE, (LONG)(style & ~WS_OVERLAPPEDWINDOW));
        SetWindowPos(gHwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                     mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED | SWP_NOOWNERZORDER);
    } else {
        SetWindowLongA(gHwnd, GWL_STYLE, (LONG)(style | WS_OVERLAPPEDWINDOW));
        SetWindowPlacement(gHwnd, &gWp);
        SetWindowPos(gHwnd, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
    }
    gFullscreen = !gFullscreen;
}

static int MakeBackUav(void)
{
    REL(gBackUav);
    REL(gBackTex);
    if (FAILED(IDXGISwapChain1_GetBuffer(gSwap, 0, &IID_ID3D11Texture2D, (void **)&gBackTex))) return 0;
    if (FAILED(ID3D11Device_CreateUnorderedAccessView(gDev, (ID3D11Resource *)gBackTex, NULL, &gBackUav))) {
        MessageBoxA(NULL, "This GPU cannot write the swap chain from a compute shader.", "solargpu", MB_OK | MB_ICONERROR);
        return 0;
    }
    return 1;
}

static int CreateSwap(void)
{
    IDXGIFactory2 *f2 = NULL;
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory2, (void **)&f2))) return 0;
    DXGI_SWAP_CHAIN_DESC1 sd;
    memset(&sd, 0, sizeof sd);
    sd.Width = (UINT)gWinW;
    sd.Height = (UINT)gWinH;
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_UNORDERED_ACCESS | DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    HRESULT hr = IDXGIFactory2_CreateSwapChainForHwnd(f2, (IUnknown *)gDev, gHwnd, &sd, NULL, NULL, &gSwap);
    IDXGIFactory2_MakeWindowAssociation(f2, gHwnd, DXGI_MWA_NO_ALT_ENTER);
    REL(f2);
    return SUCCEEDED(hr) && MakeBackUav();
}

static void FreeOverlay(void)
{
    REL(gSOverlay);
    REL(gTOverlay);
    if (gDc) { DeleteDC(gDc); gDc = NULL; }
    if (gDib) { DeleteObject(gDib); gDib = NULL; }
    if (gFont) { DeleteObject(gFont); gFont = NULL; }
    gDibBits = NULL;
    gNPrev = gNCur = 0;
}

static int CreateOverlay(int w, int h)
{
    FreeOverlay();
    BITMAPINFO bi;
    memset(&bi, 0, sizeof bi);
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    gDc = CreateCompatibleDC(NULL);
    gDib = CreateDIBSection(gDc, &bi, DIB_RGB_COLORS, (void **)&gDibBits, NULL, 0);
    if (!gDib || !gDibBits) return 0;
    memset(gDibBits, 0, (size_t)w * (size_t)h * 4);
    SelectObject(gDc, gDib);
    int fh = gWinH / 52;
    if (fh < 14) fh = 14;
    if (fh > 30) fh = 30;
    gFont = CreateFontA(-fh, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Segoe UI");
    SelectObject(gDc, gFont);
    SetBkMode(gDc, TRANSPARENT);
    gDibW = w;
    gDibH = h;
    D3D11_TEXTURE2D_DESC d;
    memset(&d, 0, sizeof d);
    d.Width = (UINT)w; d.Height = (UINT)h; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sr;
    sr.pSysMem = gDibBits;
    sr.SysMemPitch = (UINT)(w * 4);
    sr.SysMemSlicePitch = 0;
    if (FAILED(ID3D11Device_CreateTexture2D(gDev, &d, &sr, &gTOverlay))) return 0;
    if (FAILED(ID3D11Device_CreateShaderResourceView(gDev, (ID3D11Resource *)gTOverlay, NULL, &gSOverlay))) return 0;
    return 1;
}

static void OverlayBegin(void)
{
    for (int i = 0; i < gNPrev; i++) {
        const Rect *r = &gRectPrev[i];
        for (int y = r->y0; y < r->y1; y++) memset(gDibBits + ((size_t)y * (size_t)gDibW + (size_t)r->x0) * 4, 0, (size_t)(r->x1 - r->x0) * 4);
    }
    gNCur = 0;
}

static int OverlayText(int x, int y, const char *s)
{
    int len = (int)strlen(s);
    SIZE sz;
    GetTextExtentPoint32A(gDc, s, len, &sz);
    Rect r = { x - 3, y - 3, x + sz.cx + 4, y + sz.cy + 4 };
    if (r.x0 < 0 || r.y0 < 0 || r.x1 > gDibW || r.y1 > gDibH || gNCur >= 48) return 0;
    for (int i = 0; i < gNCur; i++) {
        const Rect *o = &gRectCur[i];
        if (r.x0 < o->x1 && r.x1 > o->x0 && r.y0 < o->y1 && r.y1 > o->y0) return 0;
    }
    SetTextColor(gDc, RGB(0, 255, 0));
    TextOutA(gDc, x + 1, y + 1, s, len);
    SetTextColor(gDc, RGB(255, 0, 0));
    TextOutA(gDc, x, y, s, len);
    gRectCur[gNCur++] = r;
    return 1;
}

static void OverlayEnd(void)
{
    GdiFlush();
    for (int pass = 0; pass < 2; pass++) {
        const Rect *rs = pass ? gRectCur : gRectPrev;
        int n = pass ? gNCur : gNPrev;
        for (int i = 0; i < n; i++) {
            D3D11_BOX box;
            box.left = (UINT)rs[i].x0; box.right = (UINT)rs[i].x1; box.top = (UINT)rs[i].y0; box.bottom = (UINT)rs[i].y1;
            box.front = 0; box.back = 1;
            const uint8_t *src = gDibBits + ((size_t)rs[i].y0 * (size_t)gDibW + (size_t)rs[i].x0) * 4;
            ID3D11DeviceContext_UpdateSubresource(gCtx, (ID3D11Resource *)gTOverlay, 0, &box, src, (UINT)(gDibW * 4), 0);
        }
    }
    memcpy(gRectPrev, gRectCur, sizeof(Rect) * (size_t)gNCur);
    gNPrev = gNCur;
}

static void WinLabel(int id)
{
    const Body *b = &gB[id];
    if (b->Rd <= 0.0f) return;
    float sx, sy, d;
    if (!Project(b->pos, &sx, &sy, &d)) return;
    if (sx < 0 || sy < 0 || sx >= (float)gW || sy >= (float)gH) return;
    if (Occluded(id)) return;
    float wpp = d * 2.0f * gCam.tanHy / (float)gH;
    float rext = b->Rd * (id == B_SATURN ? 2.3f : 1.0f);
    int rpx = (int)ceilf(rext / wpp);
    int len = (int)strlen(b->name);
    SIZE sz;
    GetTextExtentPoint32A(gDc, b->name, len, &sz);
    int x = (int)sx + rpx + 10, y = (int)sy - sz.cy / 2;
    if (!OverlayText(x, y, b->name)) OverlayText((int)sx - rpx - 10 - sz.cx, y, b->name);
}

static int RunWindow(float t0, int startFocus)
{
    SetProcessDPIAware();
    HINSTANCE hi = GetModuleHandleA(NULL);
    WNDCLASSEXA wc;
    memset(&wc, 0, sizeof wc);
    wc.cbSize = sizeof wc;
    wc.style = CS_DBLCLKS;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hi;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "solargpuWnd";
    RegisterClassExA(&wc);
    int scr = GetSystemMetrics(SM_CXSCREEN), scrH = GetSystemMetrics(SM_CYSCREEN);
    int cw = scr * 3 / 4, ch = cw * 9 / 16;
    if (ch > scrH * 17 / 20) { ch = scrH * 17 / 20; cw = ch * 16 / 9; }
    RECT r = { 0, 0, cw, ch };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    gHwnd = CreateWindowExA(0, "solargpuWnd", "Solar System - GPU ray tracer", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                            CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, NULL, NULL, hi, NULL);
    if (!gHwnd) return 1;
    RECT cr;
    GetClientRect(gHwnd, &cr);
    gWinW = cr.right;
    gWinH = cr.bottom;
    if (!CreateSwap()) return 1;
    SetMode(M_WIN);
    timeBeginPeriod(1);

    static const float kScale[6] = { 0.50f, 0.65f, 0.80f, 1.0f, 1.0f, 1.0f };
    static const int   kSsaa[6]  = { 1, 1, 1, 1, 2, 3 };
    int level = 4, slow = 0, fast = 0, hudOn = 1;
    gW = gWinW; gH = gWinH;
    if (!CreateTargets(0, 0) || !CreateOverlay(gW, gH)) return 1;

    LARGE_INTEGER freq, prev, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);
    InitView(t0, startFocus, 0);
    float fps = 60.0f;
    float helpT = 0.0f, titleT = 0.0f;

    while (gRun) {
        MSG msg;
        while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        for (int i = 0; i < gKeyN; i++) {
            int k = gKeyQ[i];
            if (k == K_FULL) ToggleFullscreen();
            else if (k == 'h' || k == 'H') hudOn = !hudOn;
            else if (HandleKey(k)) InterlockedExchange(&gRun, 0);
        }
        gKeyN = 0;
        if (!gRun) break;

        if (gWinDirty && (gWinW != gW || gWinH != gH)) {
            gWinDirty = 0;
            ID3D11DeviceContext_ClearState(gCtx);
            REL(gBackUav);
            REL(gBackTex);
            if (FAILED(IDXGISwapChain1_ResizeBuffers(gSwap, 0, (UINT)gWinW, (UINT)gWinH, DXGI_FORMAT_UNKNOWN, 0)) || !MakeBackUav()) break;
            gW = gWinW; gH = gWinH;
            if (!CreateTargets(0, 0) || !CreateOverlay(gW, gH)) break;
        }

        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - prev.QuadPart) / (double)freq.QuadPart);
        prev = now;
        if (dt > 0.2f) dt = 0.2f;
        if (dt > 1e-4f) fps += (1.0f / dt - fps) * 0.08f;
        helpT += dt;
        titleT += dt;
        if (titleT > 0.25f) {
            titleT = 0.0f;
            char title[160];
            snprintf(title, sizeof title, "Solar System - GPU ray tracer  [view %d %s  t=%.2f yr]", gFocus, kViewName[gFocus], gSimT);
            SetWindowTextA(gHwnd, title);
        }

        // dynamic resolution: keep the GPU time under the target
        if (gGpuMs > kTargetGpuMs * 1.15f) { slow++; fast = 0; }
        else if (gGpuMs > 0.0f && gGpuMs < kTargetGpuMs * 0.55f) { fast++; slow = 0; }
        else { slow = fast = 0; }
        if (slow >= 4 && level > 0) { level--; slow = 0; }
        if (fast >= 90 && level < 5) { level++; fast = 0; }
        gIW = (int)((float)gW * kScale[level] + 0.5f);
        gIH = (int)((float)gH * kScale[level] + 0.5f);
        if (gIW < 64) gIW = 64;
        if (gIH < 36) gIH = 36;

        StepWorld(dt, 0);

        // text first: the compose pass blends it
        OverlayBegin();
        if (gLabels) {
            if (gFocus == 0) {
                for (int i = B_EARTH; i <= B_NEPTUNE; i++) WinLabel(i);
                WinLabel(B_COMET);
                WinLabel(B_SUN);
            } else if (gFocus <= 8) {
                WinLabel(gFocus);
                if (gFocus == 3) WinLabel(B_MOON);
                if (gFocus == 5) for (int i = B_IO; i <= B_CALLISTO; i++) WinLabel(i);
                if (gFocus == 6) WinLabel(B_TITAN);
            } else {
                WinLabel(B_COMET);
                WinLabel(B_SUN);
            }
        }
        if (hudOn) {
            char hud[256];
            snprintf(hud, sizeof hud, "%s  |  t = %.2f yr   x%.2f%s%s  |  %dx%d at %.0f%% , %dx AA  |  GPU %.1f ms  %.0f fps",
                     kViewName[gFocus], gSimT, gUserMul, gPaused ? " (paused)" : "", gTourOn ? "   tour" : "", gW, gH,
                     kScale[level] * 100.0f, kSsaa[level] * kSsaa[level], gGpuMs, fps);
            OverlayText(14, gH - (gH / 52 > 14 ? gH / 52 : 14) - 26, hud);
            if (helpT < 12.0f)
                OverlayText(14, gH - (gH / 52 > 14 ? gH / 52 : 14) - 56, "0 system, 1-8 planets, 9 comet, T tour, drag = orbit, wheel = zoom, [ ] time, Space pause, L labels, O orbits, H hide this, F11 fullscreen, Q quit");
        }
        OverlayEnd();

        CB cb;
        int nScatter = 0;
        FillCB(&cb, gRealT, gSimT, gOrbits, gFocus == 0 ? 0.30f : 0.12f, kSsaa[level], 1, &nScatter);
        GpuTimingBegin();
        GpuScene(&cb, nScatter);
        GpuCompose(gBackUav);
        GpuTimingEnd();
        IDXGISwapChain1_Present(gSwap, 1, 0);
    }

    timeEndPeriod(1);
    ReleaseTargets();
    FreeOverlay();
    REL(gBackUav);
    REL(gBackTex);
    REL(gSwap);
    DestroyWindow(gHwnd);
    return 0;
}

// ================================================================= entry
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
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) { int f = atoi(argv[++i]); gTickMs = f > 0 ? 1000 / f : 16; }
        else if (!strcmp(argv[i], "--fly")) testFly = 1;
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            ++i;
            gModePref = !strcmp(argv[i], "half") ? M_HALF : !strcmp(argv[i], "quad") ? M_QUAD : M_SEXT;
        }
    }

    if (!InitGpu()) {
        fputs("GPU init failed (needs a Direct3D 11 GPU with compute shaders, and solargpu.hlsl next to the exe)\n", stderr);
        return 1;
    }
#ifdef WINDOWED
    (void)testFrames; (void)forceC; (void)forceR; (void)testFocus; (void)testFly;
    return RunWindow(t0, liveView);
#else
    return RunTerminal(t0, testFrames ? testFocus : liveView, testFrames, forceC, forceR, testFly);
#endif
}

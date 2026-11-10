// solar.c - terminal solar system. Win32 console, ANSI 256-color, fits the window.
// build: cl /O2 /W4 /nologo solar.c
// keys:  + / - speed   space pause   l labels   o orbits   q or esc quit
// dev:   solar.exe --frames N [--size CxR] [--t0 years]   render N frames and exit (no sleep)
//
// Scale notes: orbit radius on screen = kSunClear + a_au ^ kOrbitExp (a true-scale system will not fit a
// terminal). Orbital periods are real (Kepler periods in Earth years). Moon periods are slowed per system so
// they are watchable; ratios inside one system (the four Galilean moons) are real. The comet follows a real
// Kepler ellipse (e = 0.9), so it whips around the Sun at perihelion and its tail points away from the Sun.
// The view tilt adapts to the window so the orbits fill it. Planets on the near side of their orbit are
// drawn in front of the Sun, planets on the far side behind it.
#define _CRT_SECURE_NO_WARNINGS
#include <conio.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif

#define kPi  3.14159265f
#define kTau 6.28318531f

static const float kYearsPerSecDefault = 0.25f;
static const float kYearsPerSecMin = 0.01f;
static const float kYearsPerSecMax = 2.0f;
static const float kSpeedStep = 1.4f;
static const float kOrbitExp = 0.45f;   // screen radius = kSunClear + a_au^kOrbitExp
static const float kSunClear = 0.9f;    // keeps Mercury outside the Sun
static const float kSunR = 0.7f;        // Sun radius in the same units
static const float kCellAspect = 2.0f;  // terminal cell height : width
static const float kTiltMin = 0.30f;    // view tilt range; 1 = top-down
static const float kTiltMax = 0.90f;
static const float kMoonTilt = 0.8f;
enum { kTickMs = 33, kAsteroids = 320, kMaxMoons = 4, kMaxStars = 4000 };

// comet: Kepler ellipse, a in AU
static const float kCometA = 12.0f;
static const float kCometE = 0.9f;
static const float kCometOmega = 0.9f;  // orientation of perihelion
static const float kCometM0 = 5.9f;     // mean anomaly at t = 0: perihelion about 2.5 years in

typedef struct {
    const char *name;
    float revPerYear;       // real revolutions per Earth year
    float rCols;            // screen orbit radius around planet, in columns
    float phase;
    unsigned char color;
} Moon;

typedef struct {
    const char *name;
    float aAu, periodYr, phase;
    float rx, ry;           // disc radius in cells; rx <= 1 draws a single glyph
    char glyph;
    unsigned char color;
    unsigned char band[3];
    int banded, rings;
    float moonScale;        // visual slow-down for this planet's moons
    int nMoons;
    Moon moons[kMaxMoons];
} Planet;

static const Planet kPlanets[] = {
    { .name = "Mercury", .aAu = 0.387f, .periodYr = 0.241f, .phase = 0.6f, .rx = 0.5f, .ry = 0.5f, .glyph = 'o', .color = 250 },
    { .name = "Venus",   .aAu = 0.723f, .periodYr = 0.615f, .phase = 2.3f, .rx = 0.5f, .ry = 0.5f, .glyph = 'o', .color = 223 },
    { .name = "Earth",   .aAu = 1.000f, .periodYr = 1.000f, .phase = 4.1f, .rx = 0.5f, .ry = 0.5f, .glyph = 'O', .color = 39,
      .moonScale = 0.15f, .nMoons = 1, .moons = { { "Moon", 13.37f, 2.4f, 0.0f, 255 } } },
    { .name = "Mars",    .aAu = 1.524f, .periodYr = 1.881f, .phase = 5.4f, .rx = 0.5f, .ry = 0.5f, .glyph = 'o', .color = 160 },
    { .name = "Jupiter", .aAu = 5.203f, .periodYr = 11.86f, .phase = 1.1f, .rx = 2.4f, .ry = 1.2f, .glyph = '#', .color = 180,
      .band = { 180, 137, 180 }, .banded = 1,
      .moonScale = 0.01f, .nMoons = 4,
      .moons = { { "Io", 206.5f, 3.8f, 0.0f, 227 }, { "Europa", 102.9f, 4.9f, 1.6f, 255 },
                 { "Ganymede", 51.05f, 6.2f, 3.1f, 250 }, { "Callisto", 21.89f, 7.8f, 4.7f, 243 } } },
    { .name = "Saturn",  .aAu = 9.537f, .periodYr = 29.46f, .phase = 3.0f, .rx = 1.7f, .ry = 0.85f, .glyph = 'O', .color = 222,
      .rings = 1, .moonScale = 0.1f, .nMoons = 1, .moons = { { "Titan", 22.9f, 7.4f, 0.5f, 214 } } },
    { .name = "Uranus",  .aAu = 19.19f, .periodYr = 84.0f,  .phase = 0.2f, .rx = 0.5f, .ry = 0.5f, .glyph = 'O', .color = 117 },
    { .name = "Neptune", .aAu = 30.07f, .periodYr = 164.8f, .phase = 5.9f, .rx = 0.5f, .ry = 0.5f, .glyph = 'O', .color = 27 },
};
#define kNPlanets ((int)(sizeof kPlanets / sizeof kPlanets[0]))

typedef struct { char ch; unsigned char fg; } Cell;

static Cell *gGrid;
static int   gCols, gRows;
static volatile LONG gRun = 1;

static float gAstA[kAsteroids], gAstPh[kAsteroids];
static float gStarX[kMaxStars], gStarY[kMaxStars];

static uint32_t gSeed = 20240607u;
static float Rnd(void)
{
    gSeed = gSeed * 1664525u + 1013904223u;
    return (float)(gSeed >> 8) / 16777216.0f;
}

static BOOL WINAPI OnCtrl(DWORD type)
{
    (void)type;
    InterlockedExchange(&gRun, 0);
    return TRUE;
}

static void Put(int x, int y, char ch, unsigned char fg)
{
    if ((unsigned)x < (unsigned)gCols && (unsigned)y < (unsigned)(gRows - 1)) {
        Cell *c = &gGrid[y * gCols + x];
        c->ch = ch;
        c->fg = fg;
    }
}

static void PutF(float x, float y, char ch, unsigned char fg)
{
    Put((int)floorf(x + 0.5f), (int)floorf(y + 0.5f), ch, fg);
}

static void PutStr(int x, int y, const char *s, unsigned char fg)
{
    for (; *s; s++, x++)
        if (*s != ' ')
            Put(x, y, *s, fg);
}

// a label may only sit on empty cells or background dots, never over bodies or other labels
static int TryLabel(int x, int y, const char *s, unsigned char fg)
{
    int len = (int)strlen(s);
    if (x < 0 || x + len > gCols || y < 0 || y >= gRows - 1) return 0;
    for (int i = 0; i < len; i++) {
        char ch = gGrid[y * gCols + x + i].ch;
        if (ch && ch != '.' && ch != ',') return 0;
    }
    PutStr(x, y, s, fg);
    return 1;
}

static void PlaceLabel(int ax, int ay, int off, const char *s, unsigned char fg)
{
    int len = (int)strlen(s);
    int rx = ax + off, lx = ax - off + 1 - len;
    if (TryLabel(rx, ay, s, fg) || TryLabel(lx, ay, s, fg) ||
        TryLabel(rx, ay - 1, s, fg) || TryLabel(lx, ay - 1, s, fg) ||
        TryLabel(rx, ay + 1, s, fg) || TryLabel(lx, ay + 1, s, fg))
        return;
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

static void InitField(void)
{
    for (int i = 0; i < kAsteroids; i++) {
        gAstA[i] = 2.15f + 1.15f * Rnd();       // main belt, 2.15 - 3.3 AU
        gAstPh[i] = kTau * Rnd();
    }
    for (int i = 0; i < kMaxStars; i++) {
        gStarX[i] = Rnd();
        gStarY[i] = Rnd();
    }
}

static void DrawSun(float cx, float cy, float rx, float ry, float rt)
{
    ry *= 1.0f + 0.03f * sinf(rt * 3.0f);
    rx *= 1.0f + 0.03f * sinf(rt * 3.0f);

    // corona: flickering rays
    for (int j = 0; j < 22; j++) {
        float ang = j * kTau / 22.0f + rt * 0.15f;
        float fl = 0.5f + 0.5f * sinf(rt * 4.0f + j * 1.7f);
        float d = 1.2f + 0.3f * fl;
        PutF(cx + rx * d * cosf(ang), cy - ry * d * sinf(ang), '.', fl > 0.5f ? 214 : 130);
    }
    int bx = (int)ceilf(rx), by = (int)ceilf(ry);
    int ix = (int)floorf(cx + 0.5f), iy = (int)floorf(cy + 0.5f);
    for (int dy = -by; dy <= by; dy++) {
        for (int dx = -bx; dx <= bx; dx++) {
            float nx = dx / rx, ny = dy / ry;
            float nd = sqrtf(nx * nx + ny * ny);
            if (nd <= 1.0f) {
                char ch = nd < 0.45f ? '@' : nd < 0.75f ? '#' : '*';
                unsigned char col = nd < 0.45f ? 231 : nd < 0.75f ? 227 : 214;
                Put(ix + dx, iy + dy, ch, col);
            }
        }
    }
}

static void DrawRings(float px, float py, int front)
{
    for (float a = 0.0f; a < kTau; a += 0.12f) {
        float s = sinf(a);
        int isFront = s < 0.0f;
        if (isFront != front) continue;
        PutF(px + 4.6f * cosf(a), py - 1.15f * s, front ? '=' : '-', front ? 180 : 137);
    }
}

static void DrawPlanetDisc(const Planet *p, float px, float py)
{
    int ix = (int)floorf(px + 0.5f), iy = (int)floorf(py + 0.5f);
    if (p->rx <= 1.0f) {
        Put(ix, iy, p->glyph, p->color);
        return;
    }
    int bx = (int)ceilf(p->rx), by = (int)ceilf(p->ry);
    for (int dy = -by; dy <= by; dy++) {
        for (int dx = -bx; dx <= bx; dx++) {
            float nx = dx / p->rx, ny = dy / p->ry;
            if (nx * nx + ny * ny > 1.0f) continue;
            unsigned char col = p->color;
            if (p->banded) {
                col = p->band[(dy + 3) % 3];
                if (dx == 1 && dy == 1) col = 160;   // red spot
            }
            Put(ix + dx, iy + dy, p->glyph, col);
        }
    }
}

typedef struct { int x, y; float off; } LabelPos;

// returns label offset in columns
static float DrawPlanet(const Planet *p, float px, float py, float t, LabelPos *lp)
{
    if (p->rings) DrawRings(px, py, 0);
    DrawPlanetDisc(p, px, py);
    if (p->rings) DrawRings(px, py, 1);

    float off = p->rx > 1.0f ? p->rx : 1.0f;
    if (p->rings && off < 5.0f) off = 5.0f;
    for (int m = 0; m < p->nMoons; m++) {
        const Moon *mo = &p->moons[m];
        float phi = mo->phase + kTau * mo->revPerYear * p->moonScale * t;
        PutF(px + mo->rCols * cosf(phi), py - (mo->rCols / kCellAspect) * kMoonTilt * sinf(phi), '.', mo->color);
        if (mo->rCols > off) off = mo->rCols;
    }
    lp->x = (int)floorf(px + 0.5f);
    lp->y = (int)floorf(py + 0.5f);
    lp->off = off + 2.0f;
    return off;
}

static void Render(float t, float rt, int labels, int orbits, float speed, int paused)
{
    const int cols = gCols, rows = gRows;
    memset(gGrid, 0, sizeof(Cell) * (size_t)cols * (size_t)rows);

    float cx = (float)(cols - 1) * 0.5f;
    float cy = (float)(rows - 2) * 0.5f;

    // tilt adapts so the outermost orbit touches the window edges on both axes
    float tilt = kCellAspect * (cy - 1.0f) / (cx - 2.0f);
    if (tilt < kTiltMin) tilt = kTiltMin;
    if (tilt > kTiltMax) tilt = kTiltMax;
    float maxS = kSunClear + powf(kPlanets[kNPlanets - 1].aAu, kOrbitExp);
    float u1 = (cy - 1.0f) / (maxS * tilt);
    float u2 = (cx - 2.0f) / (maxS * kCellAspect);
    float u = u1 < u2 ? u1 : u2;
    if (u < 0.5f) u = 0.5f;
    float sunRy = u * kSunR * tilt;
    float sunRx = u * kSunR * kCellAspect;
    if (sunRy < 0.9f) sunRy = 0.9f;
    if (sunRx < 1.8f) sunRx = 1.8f;

    // stars (drawn first, everything overwrites them)
    int nStars = cols * rows / 30;
    if (nStars > kMaxStars) nStars = kMaxStars;
    for (int i = 0; i < nStars; i++) {
        int x = (int)(gStarX[i] * (float)cols);
        int y = (int)(gStarY[i] * (float)(rows - 1));
        int col = 234 + (i % 4) * 2;
        if (i % 5 == 0)
            col = 234 + (int)(9.0f * (0.5f + 0.5f * sinf(rt * (1.0f + (float)(i % 3)) + (float)i)));
        if (i % 25 == 0) Put(x, y, '+', 246);
        else Put(x, y, '.', (unsigned char)col);
    }

    // orbit rings
    if (orbits) {
        for (int i = 0; i < kNPlanets; i++) {
            float R = u * (kSunClear + powf(kPlanets[i].aAu, kOrbitExp));
            float rxo = R * kCellAspect, ryo = R * tilt;
            int n = (int)(3.0f * (rxo + ryo));
            for (int k = 0; k < n; k++) {
                float a = kTau * (float)k / (float)n;
                PutF(cx + rxo * cosf(a), cy - ryo * sinf(a), '.', 241);
            }
        }
    }

    // asteroid belt, each rock on its own Kepler period
    for (int i = 0; i < kAsteroids; i++) {
        float a = gAstA[i];
        float R = u * (kSunClear + powf(a, kOrbitExp));
        float ang = gAstPh[i] + kTau * t / (a * sqrtf(a));
        PutF(cx + R * kCellAspect * cosf(ang), cy - R * tilt * sinf(ang), (i % 3) ? '.' : ',', (i % 2) ? 241 : 245);
    }

    // comet: solve Kepler's equation for the eccentric anomaly
    float T = kCometA * sqrtf(kCometA);
    float M = fmodf(kTau * (t / T) + kCometM0, kTau);
    float E = kPi;
    for (int i = 0; i < 12; i++)
        E -= (E - kCometE * sinf(E) - M) / (1.0f - kCometE * cosf(E));
    float X0 = kCometA * (cosf(E) - kCometE);
    float Y0 = kCometA * sqrtf(1.0f - kCometE * kCometE) * sinf(E);
    float cw = cosf(kCometOmega), sw = sinf(kCometOmega);
    float X = X0 * cw - Y0 * sw, Y = X0 * sw + Y0 * cw;
    float rAu = sqrtf(X * X + Y * Y);
    float cs = u * (kSunClear + powf(rAu, kOrbitExp));
    float ux = X / rAu, uy = Y / rAu;
    float hx = cx + ux * cs * kCellAspect, hy = cy - uy * cs * tilt;
    float tailL = 16.0f * (1.0f - rAu / 7.0f);
    static const unsigned char tailCol[5] = { 255, 153, 117, 110, 67 };
    for (float k = 1.0f; k <= tailL; k += 0.5f) {
        int ci = (int)(k / tailL * 5.0f);
        PutF(hx + ux * k * kCellAspect, hy - uy * k * tilt, k < tailL * 0.35f ? ':' : '.', tailCol[ci > 4 ? 4 : ci]);
    }

    // planets: far side of the orbit first, then the Sun, then the near side
    float px[8], py[8];
    LabelPos lp[8];
    for (int i = 0; i < kNPlanets; i++) {
        float R = u * (kSunClear + powf(kPlanets[i].aAu, kOrbitExp));
        float ang = kPlanets[i].phase + kTau * t / kPlanets[i].periodYr;
        px[i] = cx + R * kCellAspect * cosf(ang);
        py[i] = cy - R * tilt * sinf(ang);
        lp[i].x = (int)floorf(px[i] + 0.5f);
        lp[i].y = (int)floorf(py[i] + 0.5f);
        lp[i].off = 3.0f;
    }
    for (int i = 0; i < kNPlanets; i++)
        if (py[i] < cy) DrawPlanet(&kPlanets[i], px[i], py[i], t, &lp[i]);
    DrawSun(cx, cy, sunRx, sunRy, rt);
    for (int i = 0; i < kNPlanets; i++)
        if (py[i] >= cy) DrawPlanet(&kPlanets[i], px[i], py[i], t, &lp[i]);

    PutF(hx, hy, '*', 231);

    if (labels) {
        int sy = (int)floorf(cy + 0.5f) + (int)ceilf(sunRy) + 2;
        int sx = (int)floorf(cx + 0.5f) - 1;
        if (!TryLabel(sx, sy, "Sun", 220)) TryLabel(sx, sy + 1, "Sun", 220);
        for (int i = 0; i < kNPlanets; i++)
            PlaceLabel(lp[i].x, lp[i].y, (int)lp[i].off, kPlanets[i].name, 247);
        PlaceLabel((int)floorf(hx + 0.5f), (int)floorf(hy + 0.5f), 2, "comet", 110);
    }

    // HUD on the last row
    char line[256];
    snprintf(line, sizeof line, " t=%8.2f yr   %5.2f yr/s%s   [+/-] speed  [space] pause  [l] labels  [o] orbits  [q] quit",
             t, speed, paused ? " (paused)" : "");
    int n = (int)strlen(line);
    if (n > cols - 1) n = cols - 1;
    for (int i = 0; i < n; i++) {
        Cell *c = &gGrid[(rows - 1) * cols + i];
        c->ch = line[i] == ' ' ? 0 : line[i];
        c->fg = 244;
    }
}

int main(int argc, char **argv)
{
    int testFrames = 0, forceC = 0, forceR = 0;
    float t0 = 0.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) testFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &forceC, &forceR);
        else if (!strcmp(argv[i], "--t0") && i + 1 < argc) t0 = (float)atof(argv[++i]);
    }

    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  oldMode = 0;
    BOOL   haveMode = GetConsoleMode(hOut, &oldMode);
    if (haveMode)
        SetConsoleMode(hOut, oldMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    SetConsoleCtrlHandler(OnCtrl, TRUE);

    InitField();
    fputs("\x1b[?1049h\x1b[?25l\x1b[2J", stdout);
    fflush(stdout);

    LARGE_INTEGER freq, prev, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&prev);

    float simT = t0, realT = 0.0f, speed = kYearsPerSecDefault;
    int labels = 1, orbits = 1, paused = 0;
    char *buf = NULL;
    size_t bufCap = 0;
    int lastC = 0, lastR = 0, frames = 0;

    while (gRun) {
        if (_kbhit()) {
            int c = _getch();
            if (c == 'q' || c == 'Q' || c == 27) break;
            else if (c == '+' || c == '=') { speed *= kSpeedStep; if (speed > kYearsPerSecMax) speed = kYearsPerSecMax; }
            else if (c == '-' || c == '_') { speed /= kSpeedStep; if (speed < kYearsPerSecMin) speed = kYearsPerSecMin; }
            else if (c == ' ') paused = !paused;
            else if (c == 'l' || c == 'L') labels = !labels;
            else if (c == 'o' || c == 'O') orbits = !orbits;
        }

        QueryPerformanceCounter(&now);
        float dt = (float)((double)(now.QuadPart - prev.QuadPart) / (double)freq.QuadPart);
        prev = now;
        if (testFrames) dt = 1.0f / 30.0f;
        if (dt > 0.25f) dt = 0.25f;

        int tc, tr;
        if (forceC > 0 && forceR > 0) { tc = forceC; tr = forceR; }
        else ConsoleSize(hOut, &tc, &tr);
        if (tc < 20) tc = 20;
        if (tr < 8) tr = 8;
        if (tc != lastC || tr != lastR) {
            free(gGrid);
            gGrid = malloc(sizeof(Cell) * (size_t)tc * (size_t)tr);
            size_t need = (size_t)tr * ((size_t)tc * 12 + 24) + 64;
            if (need > bufCap) {
                free(buf);
                buf = malloc(need);
                bufCap = need;
            }
            if (!gGrid || !buf) break;
            gCols = tc;
            gRows = tr;
            lastC = tc;
            lastR = tr;
            fputs("\x1b[2J", stdout);
        }

        if (!paused) simT += dt * speed;
        realT += dt;
        Render(simT, realT, labels, orbits, speed, paused);

        char *p = buf;
        int last = -1;
        for (int r = 0; r < gRows; r++) {
            p += sprintf(p, "\x1b[%d;1H", r + 1);
            for (int c = 0; c < gCols; c++) {
                const Cell *cell = &gGrid[r * gCols + c];
                if (cell->ch == 0) { *p++ = ' '; continue; }
                if (cell->fg != last) {
                    p += sprintf(p, "\x1b[38;5;%dm", cell->fg);
                    last = cell->fg;
                }
                *p++ = cell->ch;
            }
        }
        fwrite(buf, 1, (size_t)(p - buf), stdout);
        fflush(stdout);

        frames++;
        if (testFrames && frames >= testFrames) break;
        if (!testFrames) Sleep(kTickMs);
    }

    fputs("\x1b[0m\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
    if (haveMode)
        SetConsoleMode(hOut, oldMode);
    free(gGrid);
    free(buf);
    return 0;
}

// solargpu.hlsl - compute shaders for solargpu.exe (Direct3D 11, cs_5_0)
// Passes: CS_Render (ray traced scene, 3x3 supersampling) -> CS_Scatter (3D asteroids + comet tail particles)
//         -> 4x blur passes (bloom) -> CS_Compose (tone map) -> CS_Cells (best two-color fit per terminal cell)

#define PI  3.14159265
#define TAU 6.28318531

#define B_SUN 0
#define B_MERCURY 1
#define B_VENUS 2
#define B_EARTH 3
#define B_MARS 4
#define B_JUPITER 5
#define B_SATURN 6
#define B_URANUS 7
#define B_NEPTUNE 8
#define B_MOON 9
#define B_IO 10
#define B_EUROPA 11
#define B_GANYMEDE 12
#define B_CALLISTO 13
#define B_TITAN 14
#define B_COMET 15

cbuffer CB : register(b0)
{
    float4 camPos;        // xyz position, w = real time (s)
    float4 camFwd;        // xyz, w = simulated time (years)
    float4 camRight;      // xyz, w = tanHx
    float4 camUp;         // xyz, w = tanHy
    int4   dims;          // W, H, ssaa, cellsX
    int4   dims2;         // cellsY, sx, sy, flags (bit0 = orbit lines)
    float4 bPosR[16];     // xyz position, w display radius (0 = hidden)
    float4 bAxisSpin[16]; // xyz spin axis, w spin angle
    float4 bMisc[16];     // x true radius, y boost, z fill, w tilt
    float4 bTrig[16];     // cos(spin), sin(spin), cos(tilt), sin(tilt)
    float4 orbitP[8];     // x orbit radius, y inclination
    float4 cometP;        // xyz head, w tail intensity
    float4 cometV;        // xyz velocity direction, w tail length
    float4 misc;          // x orbit alpha, y bloom narrow, z bloom wide, w exposure
    float4 misc2;         // x asteroids, y tail particles, z pixel angle (rad)
    int4   dims3;         // output W, output H, windowed (overlay on), unused
    float4 misc3;         // x lens flare strength
};

Texture2D<float4> tHdr : register(t0);
Texture2D<float4> tBlurIn : register(t1);
Texture2D<float4> tBlurB : register(t2);
Texture2D<float4> tBlurC : register(t3);
Texture2D<float4> tPix : register(t4);
Texture2D<float>  tDepth : register(t5);
StructuredBuffer<uint> tScatter : register(t6);
Texture2D<float4> tOverlay : register(t7);

RWTexture2D<float4> uHdr : register(u0);
RWTexture2D<float>  uDepth : register(u1);
RWTexture2D<float4> uBlurOut : register(u2);
RWTexture2D<float4> uPix : register(u3);
RWStructuredBuffer<uint2> uCells : register(u4);
RWStructuredBuffer<uint> uScatter : register(u5);

// ------------------------------------------------------------------ helpers
float Sm(float a, float b, float x) { float t = saturate((x - a) / (b - a)); return t * t * (3.0 - 2.0 * t); }
float SunIntensity(float r) { float k = r / 75.0; return 1.0 / (1.0 + k * k); }
float WrapAngle(float a) { return a - TAU * floor((a + PI) / TAU); }

float Hash3(int3 p)
{
    uint3 u = asuint(p);
    uint h = u.x * 374761393u + u.y * 668265263u + u.z * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return float(h & 0xFFFFFFu) * (1.0 / 16777216.0);
}

float Noise3(float3 p)
{
    float3 f = floor(p);
    int3 i = (int3)f;
    float3 t = p - f;
    t = t * t * (3.0 - 2.0 * t);
    float x00 = lerp(Hash3(i), Hash3(i + int3(1, 0, 0)), t.x);
    float x10 = lerp(Hash3(i + int3(0, 1, 0)), Hash3(i + int3(1, 1, 0)), t.x);
    float x01 = lerp(Hash3(i + int3(0, 0, 1)), Hash3(i + int3(1, 0, 1)), t.x);
    float x11 = lerp(Hash3(i + int3(0, 1, 1)), Hash3(i + int3(1, 1, 1)), t.x);
    return lerp(lerp(x00, x10, t.y), lerp(x01, x11, t.y), t.z);
}

static int gExtra = 0;     // extra octaves / crater scales for bodies that cover many pixels

float Fbm(float3 p, int oct)
{
    oct = min(oct + gExtra, 9);
    float a = 0.5, s = 0.0, f = 1.0, tot = 0.0;
    [loop] for (int i = 0; i < oct; i++)
    {
        s += a * Noise3(p * f);
        tot += a;
        f *= 2.03;
        a *= 0.5;
    }
    return s / tot;
}

float Craters(float3 p, float freq, float seed)
{
    float3 q = float3(p.x * freq + seed, p.y * freq + seed * 0.7, p.z * freq - seed * 0.3);
    float3 fq = floor(q);
    int3 ib = (int3)fq;
    float3 fr = q - fq;
    int3 sg = int3(fr.x < 0.5 ? -1 : 1, fr.y < 0.5 ? -1 : 1, fr.z < 0.5 ? -1 : 1);
    float h = 0.0;
    [loop] for (int k = 0; k < 8; k++)
    {
        int3 c = ib + int3((k & 1) ? sg.x : 0, (k & 2) ? sg.y : 0, (k & 4) ? sg.z : 0);
        if (Hash3(c) < 0.40) continue;
        float3 cc = float3(c) + 0.25 + 0.5 * float3(Hash3(c + int3(13, 0, 0)), Hash3(c + int3(0, 29, 0)), Hash3(c + int3(0, 0, 57)));
        float r = 0.14 + 0.24 * Hash3(c + int3(5, 5, 5));
        float d = length(q - cc) / r;
        if (d < 1.0) h -= 1.0 - d * d;
        if (d > 0.8 && d < 1.4) { float e = (d - 1.0) * 5.0; h += 0.55 * exp(-e * e); }
    }
    return h;
}

float Worley2(float u, float v)
{
    float fu = floor(u), fv = floor(v);
    int iu = (int)fu, iv = (int)fv;
    float best = 9.0;
    [loop] for (int dv = -1; dv <= 1; dv++)
        [loop] for (int du = -1; du <= 1; du++)
        {
            int cu = iu + du, cv = iv + dv;
            float px = float(cu) + Hash3(int3(cu, cv, 3));
            float py = float(cv) + Hash3(int3(cu, cv, 9));
            float d = sqrt((u - px) * (u - px) + (v - py) * (v - py));
            best = min(best, d);
        }
    return best;
}

bool RaySphere(float3 o, float3 d, float3 c, float r, out float t)
{
    t = 0.0;
    float3 oc = o - c;
    float b = dot(oc, d);
    float cc = dot(oc, oc) - r * r;
    float disc = b * b - cc;
    if (disc < 0.0) return false;
    float s = sqrt(disc);
    float t0 = -b - s;
    if (t0 > 1e-4) { t = t0; return true; }
    float t1 = -b + s;
    if (t1 > 1e-4) { t = t1; return true; }
    return false;
}

// ------------------------------------------------------------------ rings and shadows
void RingProps(float rho, out float alpha, out float3 col)
{
    alpha = 0.0;
    col = 0.0;
    if (rho < 1.24 || rho > 2.27) return;
    float a;
    float3 c;
    if (rho < 1.52)      { a = 0.10 + 0.18 * (rho - 1.24) / 0.28; c = float3(0.42, 0.38, 0.33); }
    else if (rho < 1.95) { a = 0.88; c = float3(0.86, 0.78, 0.64); }
    else if (rho < 2.03) { a = 0.03; c = float3(0.50, 0.45, 0.40); }
    else                 { a = 0.58; c = float3(0.72, 0.66, 0.56); }
    if (rho > 2.205 && rho < 2.225) a = 0.04;
    float n = Noise3(float3(rho * 60.0, 0.5, 0.5)) * 0.6 + Noise3(float3(rho * 170.0, 1.5, 0.5)) * 0.4;
    a *= 0.62 + 0.5 * n;
    c *= 0.80 + 0.4 * Noise3(float3(rho * 25.0, 3.1, 0.5));
    alpha = min(a, 1.0);
    col = c;
}

bool RingHit(float3 o, float3 d, out float tOut, out float alpha, out float3 col)
{
    tOut = 0.0; alpha = 0.0; col = 0.0;
    float3 sp = bPosR[B_SATURN].xyz;
    float3 ax = bAxisSpin[B_SATURN].xyz;
    float rd = bPosR[B_SATURN].w;
    float den = dot(d, ax);
    if (abs(den) < 1e-5) return false;
    float t = dot(sp - o, ax) / den;
    if (t < 1e-3) return false;
    float3 q = o + d * t;
    float rho = length(q - sp) / rd;
    RingProps(rho, alpha, col);
    if (alpha < 0.01) return false;
    tOut = t;
    return true;
}

float SphereShadow(float3 p, float3 L, float3 c, float r)
{
    float3 oc = c - p;
    float tca = dot(oc, L);
    if (tca <= 0.0) return 1.0;
    float d2 = dot(oc, oc) - tca * tca;
    float d = sqrt(max(d2, 0.0));
    float w = 0.03 * r + tca * 0.04;
    return 0.04 + 0.96 * Sm(r - w, r + w, d);
}

float Shadow(float3 p, float3 L, int self)
{
    float s = 1.0;
    [loop] for (int i = 1; i < 16; i++)
    {
        float r = bPosR[i].w;
        if (i == self || r <= 0.0) continue;
        s = min(s, SphereShadow(p, L, bPosR[i].xyz, r));
    }
    float t, a;
    float3 c;
    if (RingHit(p, L, t, a, c)) s *= 1.0 - 0.9 * a;
    return s;
}

// ------------------------------------------------------------------ surfaces
struct Surf { float3 alb; float spec; float3 emis; float atmo; };

float3 RotY(float3 v, float a)
{
    float c = cos(a), s = sin(a);
    return float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

float3 ToObj(int id, float3 n)
{
    float4 t = bTrig[id];      // cs ss ct st
    float3 a = float3(n.x * t.z + n.y * t.w, -n.x * t.w + n.y * t.z, n.z);
    return float3(a.x * t.x - a.z * t.y, a.y, a.x * t.y + a.z * t.x);
}

float AngDist(float3 n, float lat0, float lon0)
{
    float3 v = float3(cos(lat0) * cos(lon0), sin(lat0), cos(lat0) * sin(lon0));
    return acos(clamp(dot(n, v), -1.0, 1.0));
}

float EuropaCracks(float3 n)
{
    float c1 = abs(Noise3(n * 9.0) - 0.5);
    float c2 = abs(Noise3(n * 17.0 + float3(3.0, 0.0, 0.0)) - 0.5);
    return 1.0 - Sm(0.0, 0.035, min(c1, c2));
}

float Height0(int id, float3 n)
{
    switch (id)
    {
    case B_MOON:
        return 0.5 * Craters(n, 5.0, 1.0) + 0.3 * Craters(n, 12.0, 4.0) + 0.18 * Craters(n, 27.0, 9.0) + 0.25 * Fbm(n * 6.0, 3);
    case B_MERCURY:
        return 0.6 * Craters(n, 6.0, 2.0) + 0.35 * Craters(n, 14.0, 7.0) + 0.2 * Craters(n, 30.0, 3.0) + 0.2 * Fbm(n * 8.0, 3);
    case B_MARS:
    {
        float a = AngDist(n, 0.35, 3.8);
        return 0.35 * Craters(n, 6.0, 5.0) + 0.2 * Craters(n, 15.0, 8.0)
             + 0.5 * Fbm(n * 3.2 + float3(11.0, 3.0, 5.0), 4) + 1.6 * exp(-(a / 0.12) * (a / 0.12));
    }
    case B_CALLISTO:
        return 0.6 * Craters(n, 7.0, 6.0) + 0.4 * Craters(n, 16.0, 2.0) + 0.25 * Craters(n, 34.0, 1.0);
    case B_GANYMEDE:
        return 0.3 * Craters(n, 6.0, 3.0) + 0.35 * Fbm(n * 10.0, 3) + 0.12 * sin(Noise3(n * 5.0) * 28.0);
    case B_IO:
        return 0.3 * Fbm(n * 5.0, 3);
    case B_EUROPA:
        return -0.7 * EuropaCracks(n);
    case B_COMET:
        return 1.4 * Fbm(n * 5.0, 4) + 0.4 * Craters(n, 10.0, 3.0);
    case B_EARTH:
    {
        float land = Fbm(n * 2.3 + float3(3.1, 1.7, 5.3), 5);
        return land > 0.5 ? 0.9 * Sm(0.5, 0.8, land) * Fbm(n * 9.0, 3) : 0.0;
    }
    }
    return 0.0;
}

float Height(int id, float3 n)
{
    float h = Height0(id, n);
    if (id == B_MOON || id == B_MERCURY || id == B_CALLISTO || id == B_MARS || id == B_GANYMEDE)
    {
        if (gExtra >= 1) h += 0.10 * Craters(n, 60.0, 11.0);
        if (gExtra >= 2) h += 0.06 * Craters(n, 130.0, 5.0);
        if (gExtra >= 3) h += 0.04 * Craters(n, 280.0, 2.0);
    }
    return h;
}

float BumpStrength(int id)
{
    switch (id)
    {
    case B_MERCURY: return 0.055;
    case B_EARTH: return 0.05;
    case B_MARS: return 0.045;
    case B_MOON: return 0.05;
    case B_IO: return 0.04;
    case B_EUROPA: return 0.06;
    case B_GANYMEDE: return 0.045;
    case B_CALLISTO: return 0.055;
    case B_COMET: return 0.07;
    }
    return 0.0;
}

float3 Bump(int id, float3 n)
{
    float k = BumpStrength(id);
    if (k <= 0.0 || bMisc[id].z > 0.85) return n;
    float3 t1 = abs(n.y) < 0.95 ? normalize(cross(n, float3(0, 1, 0))) : normalize(cross(n, float3(1, 0, 0)));
    float3 t2 = cross(n, t1);
    float e = 0.012 / (1.0 + float(gExtra));
    float h0 = Height(id, ToObj(id, n));
    float h1 = Height(id, ToObj(id, normalize(n + t1 * e)));
    float h2 = Height(id, ToObj(id, normalize(n + t2 * e)));
    float3 g = t1 * ((h1 - h0) / e) + t2 * ((h2 - h0) / e);
    return normalize(n - g * k);
}

Surf Tex(int id, float3 n, float3 Lo, float rt)
{
    Surf s;
    s.alb = float3(0.5, 0.5, 0.5);
    s.spec = 0.0;
    s.emis = 0.0;
    s.atmo = 0.0;
    float lat = asin(clamp(n.y, -1.0, 1.0));
    float lon = atan2(n.z, n.x);
    float la = abs(lat);
    switch (id)
    {
    case B_MERCURY:
    {
        float f = Fbm(n * 7.0, 5);
        float g = 0.26 + 0.40 * f;
        float cr = Craters(n, 14.0, 7.0);
        g *= cr < -0.35 ? 0.78 : 1.0;
        g *= 1.0 + 0.35 * Sm(0.35, 0.9, cr);
        float ray = Sm(0.84, 0.92, Noise3(n * 18.0));
        s.alb = float3(g * 1.03 + ray * 0.12, g * 0.97 + ray * 0.11, g * 0.92 + ray * 0.10);
        break;
    }
    case B_VENUS:
    {
        float warp = Fbm(n * 2.5, 3);
        float band = Fbm(float3(n.x * 2.0 + warp * 1.4, n.y * 6.0 + warp * 1.6, n.z * 2.0), 5);
        float fine = Fbm(float3(n.x * 9.0 + warp * 2.0, n.y * 16.0, n.z * 9.0), 3);
        s.alb = lerp(float3(0.97, 0.85, 0.56), float3(0.80, 0.55, 0.26), band * 0.8 + fine * 0.35);
        s.atmo = 0.9;
        break;
    }
    case B_EARTH:
    {
        float land = Fbm(n * 2.3 + float3(3.1, 1.7, 5.3), 5);
        float3 col;
        float spec;
        float cityMask = 0.0;
        if (land > 0.5)
        {
            float h = saturate((land - 0.5) * 5.0);
            float dry = exp(-((la - 0.5) * (la - 0.5)) / 0.05) * Sm(0.40, 0.62, Fbm(n * 5.0, 3));
            col = lerp(float3(0.08, 0.32, 0.09), float3(0.22, 0.46, 0.14), Noise3(n * 9.0));
            col = lerp(col, float3(0.62, 0.50, 0.28), dry);
            col = lerp(col, float3(0.38, 0.34, 0.30), Sm(0.55, 1.0, h) * 0.75);
            col *= 0.85 + 0.3 * Fbm(n * 22.0, 3);
            spec = 0.03;
            cityMask = 1.0;
        }
        else
        {
            float d = saturate((0.5 - land) * 4.0);
            col = lerp(float3(0.08, 0.34, 0.66), float3(0.01, 0.06, 0.28), d);
            col = lerp(col, float3(0.10, 0.45, 0.62), Sm(0.46, 0.5, land) * (1.0 - d) * 0.8);
            spec = 0.9;
        }
        float ice = Sm(1.26, 1.36, la + 0.14 * (Fbm(n * 6.0, 3) - 0.5));
        col = lerp(col, float3(0.92, 0.94, 0.97), ice);
        spec = lerp(spec, 0.25, ice);
        cityMask *= 1.0 - ice;
        float3 cn = RotY(n, rt * 0.03);
        float cl = Sm(0.50, 0.72, Fbm(cn * 3.4 + float3(7.0, 1.0, 3.0), 5));
        float3 sh = normalize(n - (Lo - n * dot(Lo, n)) * 0.035);
        float3 cs = RotY(sh, rt * 0.03);
        float clS = Sm(0.50, 0.72, Fbm(cs * 3.4 + float3(7.0, 1.0, 3.0), 4));
        col *= 1.0 - 0.5 * clS * (1.0 - cl);
        col = lerp(col, float3(0.96, 0.96, 0.98), cl * 0.93);
        spec *= 1.0 - cl;
        if (cityMask > 0.5 && Noise3(n * 70.0) > 0.78 && Fbm(n * 7.0, 3) > 0.45)
            s.emis = float3(1.0, 0.72, 0.30) * 1.1 * (1.0 - cl);
        s.alb = col; s.spec = spec; s.atmo = 1.0;
        break;
    }
    case B_MARS:
    {
        float f = Fbm(n * 3.2 + float3(11.0, 3.0, 5.0), 5);
        float3 col = lerp(float3(0.60, 0.27, 0.13), float3(0.80, 0.47, 0.26), f);
        float d = Fbm(n * 2.0 + float3(1.0, 9.0, 4.0), 4);
        col = lerp(col, float3(0.26, 0.13, 0.09), Sm(0.52, 0.66, d) * 0.8);
        float dl = WrapAngle(lon - 5.2);
        float canyon = abs(dl) < 0.55 ? 1.0 - Sm(0.0, 0.04, abs(lat + 0.1 + 0.04 * sin(dl * 10.0))) : 0.0;
        col = lerp(col, float3(0.22, 0.10, 0.07), canyon * 0.85 * (1.0 - Sm(0.35, 0.55, abs(dl))));
        float oly = AngDist(n, 0.35, 3.8);
        col = lerp(col, float3(0.85, 0.62, 0.48), exp(-(oly / 0.11) * (oly / 0.11)) * 0.5);
        col = lerp(col, float3(0.93, 0.93, 0.96), Sm(1.38, 1.46, la + 0.08 * f));
        s.alb = col; s.atmo = 0.25;
        break;
    }
    case B_JUPITER:
    {
        float turb = Fbm(float3(n.x * 4.0, n.y * 14.0, n.z * 4.0), 5);
        float latw = lat + 0.10 * (Fbm(float3(n.x * 5.0, n.y * 9.0, n.z * 5.0), 3) - 0.5);
        float band = sin(latw * 10.5 + turb * 2.4);
        float3 col = lerp(float3(0.58, 0.36, 0.24), float3(0.95, 0.87, 0.73), 0.5 + 0.5 * band);
        float belt = Sm(0.65, 0.95, sin(latw * 21.0 + turb * 3.5));
        col = lerp(col, float3(0.42, 0.25, 0.18), belt * 0.55);
        float streak = Fbm(float3(lon * 2.0, lat * 40.0, 1.0), 3);
        col *= 0.88 + 0.24 * streak;
        float w = Worley2(lon * 4.0, lat * 11.0);
        col = lerp(col, float3(0.97, 0.94, 0.88), (1.0 - Sm(0.0, 0.16, w)) * 0.55 * Sm(0.55, 0.75, turb));
        float dl = WrapAngle(lon - 1.0);
        float e = (dl / 0.30) * (dl / 0.30) + ((lat + 0.40) / 0.15) * ((lat + 0.40) / 0.15);
        if (e < 1.5)
        {
            float swirl = 0.5 + 0.5 * sin(atan2(lat + 0.40, dl * 0.5) * 3.0 + e * 9.0);
            float3 spot = lerp(float3(0.78, 0.32, 0.20), float3(0.90, 0.55, 0.38), swirl * 0.6);
            col = lerp(col, spot, (e < 1.0 ? sqrt(1.0 - e) * 0.95 : 0.0) + (e >= 1.0 ? (1.5 - e) * 0.3 : 0.0));
        }
        s.alb = col; s.atmo = 0.25;
        break;
    }
    case B_SATURN:
    {
        float turb = Fbm(float3(n.x * 3.0, n.y * 12.0, n.z * 3.0), 4);
        float t = 0.5 + 0.5 * sin(lat * 12.0 + turb * 1.6);
        float3 col = lerp(float3(0.74, 0.63, 0.43), float3(0.93, 0.85, 0.64), t);
        col *= 0.92 + 0.16 * Fbm(float3(lon * 2.0, lat * 36.0, 2.0), 3);
        col *= 1.0 - 0.25 * Sm(0.9, 1.5, la);
        if (lat > 1.35)
        {
            float th = WrapAngle(lon) + PI;
            float ph = fmod(th, PI / 3.0) - PI / 6.0;
            float rp = PI * 0.5 - lat;
            float hex = 0.17 / cos(ph);
            col = lerp(col, float3(0.42, 0.48, 0.55), (1.0 - Sm(hex - 0.015, hex + 0.015, rp)) * 0.75);
        }
        s.alb = col; s.atmo = 0.3;
        break;
    }
    case B_URANUS:
    {
        float f = 0.92 + 0.08 * sin(lat * 9.0 + Fbm(n * 3.0, 3) * 2.0);
        float3 col = float3(0.55, 0.82, 0.86) * f;
        col = lerp(col, float3(0.75, 0.92, 0.95), Sm(0.9, 1.4, la) * 0.5);
        s.alb = col; s.atmo = 0.8;
        break;
    }
    case B_NEPTUNE:
    {
        float f = Fbm(float3(n.x * 2.0, n.y * 9.0, n.z * 2.0), 4);
        float3 col = lerp(float3(0.10, 0.20, 0.70), float3(0.26, 0.46, 0.96), f);
        float streak = Sm(0.70, 0.78, Fbm(float3(n.x * 3.0, n.y * 18.0 + 3.0, n.z * 3.0), 3));
        col = lerp(col, float3(0.85, 0.90, 1.0), streak * 0.55);
        float dl = WrapAngle(lon - 2.2);
        float e = (dl / 0.32) * (dl / 0.32) + ((lat + 0.35) / 0.17) * ((lat + 0.35) / 0.17);
        if (e < 1.0) col = lerp(col, float3(0.04, 0.08, 0.35), sqrt(1.0 - e) * 0.8);
        s.alb = col; s.atmo = 0.8;
        break;
    }
    case B_MOON:
    {
        float f = Fbm(n * 6.0, 5);
        float g = 0.30 + 0.42 * f;
        float maria = Sm(0.50, 0.62, Fbm(n * 1.6 + float3(4.0, 2.0, 8.0), 3));
        g *= 1.0 - 0.40 * maria;
        float cr = Craters(n, 12.0, 4.0);
        g *= 1.0 + 0.25 * Sm(0.3, 0.9, cr) - 0.12 * saturate(-cr);
        float a = AngDist(n, -0.75, 0.5);
        if (a < 1.2)
        {
            float3 v0 = float3(cos(-0.75) * cos(0.5), sin(-0.75), cos(-0.75) * sin(0.5));
            float3 e1 = normalize(cross(v0, float3(0, 1, 0))), e2 = cross(v0, e1);
            float az = atan2(dot(n, e2), dot(n, e1));
            float ray = pow(0.5 + 0.5 * sin(az * 11.0 + 3.0 * Noise3(float3(az * 2.0, 0.5, 0.5))), 6.0) * (1.0 - Sm(0.08, 1.1, a));
            g += 0.35 * ray + (a < 0.06 ? 0.25 : 0.0);
        }
        s.alb = float3(g, g * 0.98, g * 0.95);
        break;
    }
    case B_IO:
    {
        float sp = Noise3(n * 9.0);
        float3 col = lerp(float3(0.90, 0.80, 0.30), float3(0.82, 0.55, 0.20), Fbm(n * 4.0, 3));
        col = lerp(col, float3(0.15, 0.10, 0.08), Sm(0.72, 0.80, sp));
        float ring = Sm(0.60, 0.66, sp) * (1.0 - Sm(0.66, 0.72, sp));
        col = lerp(col, float3(0.85, 0.25, 0.12), ring * 0.7);
        s.alb = col;
        break;
    }
    case B_EUROPA:
    {
        float3 col = lerp(float3(0.88, 0.84, 0.76), float3(0.76, 0.70, 0.62), Fbm(n * 5.0, 3));
        col = lerp(col, float3(0.52, 0.32, 0.22), EuropaCracks(n) * 0.8);
        s.alb = col;
        break;
    }
    case B_GANYMEDE:
    {
        float f = Fbm(n * 5.0, 4);
        float3 col = lerp(float3(0.32, 0.29, 0.26), float3(0.62, 0.58, 0.52), Sm(0.45, 0.65, f));
        col *= 1.0 + 0.3 * Sm(0.3, 0.9, Craters(n, 6.0, 3.0));
        s.alb = col;
        break;
    }
    case B_CALLISTO:
    {
        float f = Fbm(n * 7.0, 4);
        float g = 0.18 + 0.22 * f;
        float cr = Craters(n, 16.0, 2.0);
        float bright = Sm(0.84, 0.92, Noise3(n * 20.0)) + 0.35 * Sm(0.3, 0.9, cr);
        s.alb = float3(g + bright * 0.4, g * 0.95 + bright * 0.4, g * 0.9 + bright * 0.38);
        break;
    }
    case B_TITAN:
    {
        float f = Fbm(n * 3.0, 3);
        s.alb = lerp(float3(0.86, 0.55, 0.20), float3(0.72, 0.42, 0.14), f);
        s.atmo = 1.0;
        break;
    }
    case B_COMET:
    {
        float f = Fbm(n * 5.0, 4);
        float g = 0.08 + 0.18 * f;
        s.alb = float3(g * 1.1, g, g * 0.9);
        break;
    }
    }
    return s;
}

float3 Shade(int id, float3 p, float3 n, float3 rd, float rt)
{
    float3 L = normalize(-p);
    Surf s = Tex(id, ToObj(id, n), ToObj(id, L), rt);
    float3 nb = Bump(id, n);
    float boost = bMisc[id].y;
    float ex = bMisc[id].z;
    float ndd = dot(nb, L);
    float wrap = 0.12 * s.atmo;
    float diff = saturate((ndd + wrap) / (1.0 + wrap));
    float fillL = ex * 0.55 * max(dot(nb, -rd), 0.0);
    diff = max(diff, fillL);
    float sh = 0.0;
    if (diff > 0.0) sh = ex > 0.3 ? 1.0 : Shadow(p + n * 1e-3, L, id);
    float I = SunIntensity(length(p));
    float lit = diff * sh * I;
    float3 c = s.alb * (lit + (boost > 1.05 ? 0.06 : 0.010));
    if (lit > 0.0 && s.spec > 0.0)
    {
        float3 H = normalize(L - rd);
        float sp = pow(max(dot(nb, H), 0.0), 60.0) * s.spec * sh * I;
        c += float3(sp, sp, sp * 0.95);
    }
    c += s.emis * (1.0 - saturate(diff * sh * 4.0));
    return c * boost;
}

float3 SunColor(float3 n, float3 rd, float rt)
{
    float mu = saturate(-dot(n, rd));
    float limb = 0.5 + 0.5 * sqrt(mu);
    float3 p = n * 8.0 + float3(rt * 0.15, 0.0, rt * 0.10);
    float g = Fbm(p, 5);
    float cells = 1.0 - Worley2(n.x * 22.0 + n.z * 17.0 + rt * 0.05, n.y * 22.0 + n.z * 9.0);
    float spots = Sm(0.60, 0.70, Fbm(n * 3.0 + float3(0.0, rt * 0.02, 5.0), 3));
    float3 c = lerp(float3(1.0, 0.50, 0.12), float3(1.0, 0.92, 0.58), saturate(mu * 1.1 + (g - 0.5) * 0.7 + (cells - 0.5) * 0.25));
    c *= (0.86 + 0.30 * g + 0.12 * cells) * (1.10 + 0.40 * limb);
    return c * (1.0 - 0.45 * spots);
}

// ------------------------------------------------------------------ atmospheres (ray-marched single scattering)
bool AtmoParams(int id, out float3 beta, out float str)
{
    beta = 0.0; str = 0.0;
    switch (id)
    {
    case B_VENUS:   beta = float3(1.00, 0.80, 0.40); str = 1.4; return true;
    case B_EARTH:   beta = float3(0.30, 0.62, 1.35); str = 1.0; return true;
    case B_MARS:    beta = float3(0.90, 0.60, 0.40); str = 0.25; return true;
    case B_JUPITER: beta = float3(0.90, 0.75, 0.55); str = 0.22; return true;
    case B_SATURN:  beta = float3(0.95, 0.85, 0.60); str = 0.25; return true;
    case B_URANUS:  beta = float3(0.40, 0.85, 1.00); str = 0.8; return true;
    case B_NEPTUNE: beta = float3(0.25, 0.45, 1.20); str = 0.8; return true;
    case B_TITAN:   beta = float3(1.00, 0.55, 0.20); str = 1.5; return true;
    }
    return false;
}

void ApplyAtmospheres(float3 ro, float3 rd, float tBest, inout float3 col)
{
    [loop] for (int a = 1; a <= B_TITAN; a++)
    {
        float R = bPosR[a].w;
        if (R <= 0.0) continue;
        float3 beta;
        float str;
        if (!AtmoParams(a, beta, str)) continue;
        float3 c = bPosR[a].xyz;
        float3 oc = c - ro;
        float tca = dot(oc, rd);
        float shell = R * 1.14;
        float d2 = dot(oc, oc) - tca * tca;
        if (d2 >= shell * shell) continue;
        float hh = sqrt(shell * shell - d2);
        float t0 = max(tca - hh, 0.0), t1 = tca + hh;
        if (t1 <= 0.0) continue;
        float tp = 1e30, tmp;
        if (RaySphere(ro, rd, c, R, tmp)) tp = tmp;
        float tEnd = min(t1, min(tp, tBest));
        if (tEnd <= t0) continue;
        const int N = 12;
        float dl = (tEnd - t0) / float(N);
        float H = R * 0.035;
        float3 insc = 0.0;
        float3 viewT = 1.0;
        [loop] for (int k = 0; k < N; k++)
        {
            float t = t0 + (float(k) + 0.5) * dl;
            float3 p = ro + rd * t;
            float3 v = p - c;
            float r = length(v);
            float dens = exp(-max(r - R, 0.0) / H);
            float3 up = v / r;
            float3 L = normalize(-p);
            float mu = dot(up, L);
            float vis = SphereShadow(p, L, c, R);
            float3 tauS = beta * dens * 0.30 / (saturate(mu) + 0.08);
            float3 sunT = exp(-tauS);
            float cv = dot(rd, L);
            float phase = 0.75 * (1.0 + cv * cv) + 0.5 * pow(saturate(cv), 8.0);
            float3 tauStep = beta * (dens * dl / H) * 0.05;
            insc += tauStep * phase * 0.62 * sunT * vis * SunIntensity(length(p)) * viewT * str;
            viewT *= exp(-tauStep * min(str, 1.0));
        }
        col = col * viewT + insc * bMisc[a].y;
    }
}

// ------------------------------------------------------------------ sky
float3 Stars(float3 rd)
{
    float3 acc = 0.0;
    float sig = misc2.z * 0.5;
    float rt = camPos.w;
    [unroll] for (int layer = 0; layer < 3; layer++)
    {
        float scale = layer == 0 ? 38.0 : (layer == 1 ? 90.0 : 190.0);
        float dens = layer == 0 ? 0.012 : (layer == 1 ? 0.010 : 0.006);
        float3 p = rd * scale;
        float3 fp = floor(p);
        int3 ib = (int3)fp;
        float3 fr = p - fp;
        int3 sg = int3(fr.x < 0.5 ? -1 : 1, fr.y < 0.5 ? -1 : 1, fr.z < 0.5 ? -1 : 1);
        [unroll] for (int k = 0; k < 8; k++)
        {
            int3 c = ib + int3((k & 1) ? sg.x : 0, (k & 2) ? sg.y : 0, (k & 4) ? sg.z : 0);
            float h = Hash3(c + int3(layer * 131, 7, 3));
            if (h > dens) continue;
            float3 sp = float3(c) + float3(Hash3(c + int3(17, 0, 0)), Hash3(c + int3(0, 31, 0)), Hash3(c + int3(0, 0, 47)));
            float3 dv = rd - normalize(sp);
            float d2 = dot(dv, dv);
            float hb = Hash3(c + int3(5, 9, 13));
            float b = pow(hb, 9.0) * 1.1 + 0.012;
            float tw = 0.85 + 0.15 * sin(rt * (1.0 + 4.0 * hb) + hb * 60.0);
            float3 tint = lerp(float3(0.75, 0.85, 1.0), float3(1.0, 0.82, 0.65), Hash3(c + int3(23, 29, 41)));
            float g = exp(-d2 / (sig * sig));
            acc += tint * (b * tw) * g;
            if (b > 0.6)
            {
                float dx = dot(dv, camRight.xyz), dy = dot(dv, camUp.xyz);
                float sp2 = exp(-abs(dx) / (sig * 7.0)) * exp(-dy * dy / (sig * sig * 0.5))
                          + exp(-abs(dy) / (sig * 7.0)) * exp(-dx * dx / (sig * sig * 0.5));
                acc += tint * (b * tw) * 0.30 * sp2;
            }
        }
    }
    return acc;
}

float3 Sky(float3 rd)
{
    float3 col = Stars(rd);
    float b = dot(rd, float3(0.2504, 0.8813, 0.4006));
    float band = exp(-b * b * 14.0);
    if (band > 0.03)
    {
        float cloud = Fbm(rd * 3.5, 4);
        float fine = Fbm(rd * 14.0, 3);
        float dust = Sm(0.35, 0.65, Fbm(rd * 6.0 + float3(5.0, 2.0, 9.0), 3));
        float v = band * (0.35 + 1.1 * cloud) * (0.55 + 0.9 * fine) * (1.0 - 0.55 * dust) * 0.085;
        col += lerp(float3(0.55, 0.62, 1.0), float3(1.0, 0.78, 0.55), saturate(cloud * 1.4 - 0.3)) * v;
    }
    float nb = Fbm(rd * 2.2 + float3(9.0, 4.0, 1.0), 4);
    col += float3(0.30, 0.10, 0.38) * Sm(0.55, 0.85, nb) * 0.030;
    col += float3(0.05, 0.14, 0.34) * Sm(0.50, 0.80, Fbm(rd * 1.7 + float3(2.0, 7.0, 3.0), 3)) * 0.020;
    return col;
}

// ------------------------------------------------------------------ the scene
float3 Trace(float3 ro, float3 rd, out float depthOut)
{
    float rt = camPos.w;
    float tBest = 1e30;
    int hit = -1;
    [loop] for (int i = 0; i < 16; i++)
    {
        float R = bPosR[i].w;
        if (R <= 0.0) continue;
        float t;
        if (RaySphere(ro, rd, bPosR[i].xyz, R, t) && t < tBest) { tBest = t; hit = i; }
    }
    float3 col = 0.0;
    float depth = 1e30;
    if (hit == B_SUN)
    {
        float3 p = ro + rd * tBest;
        col = SunColor(normalize(p - bPosR[B_SUN].xyz), rd, rt);
        depth = tBest;
    }
    else if (hit > 0)
    {
        float3 p = ro + rd * tBest;
        float rpx = bPosR[hit].w / max(tBest * misc2.z, 1e-6);
        gExtra = (int)clamp(floor(log2(max(rpx, 1.0) / 70.0)), 0.0, 3.0);
        col = Shade(hit, p, normalize(p - bPosR[hit].xyz), rd, rt);
        gExtra = 0;
        depth = tBest;
    }
    else
    {
        col = Sky(rd);
    }
    {
        float tr, a;
        float3 rc;
        if (RingHit(ro, rd, tr, a, rc) && tr < tBest)
        {
            float3 q = ro + rd * tr;
            float3 L = normalize(-q);
            float sh = Shadow(q, L, -1);
            float3 ax = bAxisSpin[B_SATURN].xyz;
            float ndl = dot(ax, L);
            float lit = (0.55 + 0.60 * abs(ndl)) * sh * SunIntensity(length(q));
            if ((ndl > 0.0) != (dot(ax, -rd) > 0.0)) lit *= 0.55;
            col = lerp(col, rc * (lit + 0.01), a);
            if (a > 0.5) depth = tr;
        }
    }
    ApplyAtmospheres(ro, rd, tBest, col);
    {   // solar glow: corona, halo and prominences
        float3 oc = -ro;
        float ocl2 = dot(oc, oc);
        float Rs = bPosR[B_SUN].w;
        Rs = bMisc[B_SUN].x;
        float tca = dot(oc, rd);
        if (tca > 0.0 && ocl2 > Rs * Rs * 1.05 && !(hit > 0 && tBest < tca))
        {
            float d2 = ocl2 - tca * tca;
            float tanA = sqrt(max(d2, 0.0)) / tca;
            float tanS = Rs / sqrt(ocl2 - Rs * Rs);
            float s = tanA / tanS;
            if (s > 0.98)
            {
                float q = (s - 1.0) * 2.0;
                float g = 0.45 / (1.0 + q * q) + 0.07 / (s * s);
                if (s < 2.0)
                {
                    float3 nq = normalize(ro + rd * tca);
                    float fl = Sm(0.56, 0.80, Fbm(nq * 5.0 + float3(rt * 0.25, 0.0, rt * 0.1), 3));
                    g += 0.55 * fl * exp(-(s - 1.0) * 3.2);
                }
                col += float3(1.0, 0.70, 0.32) * g;
            }
        }
    }
    if ((dims2.w & 1) != 0)
    {
        float3 lineCol = float3(0.30, 0.42, 0.75);
        [loop] for (int k = 0; k < 8; k++)
        {
            float R = orbitP[k].x, inc = orbitP[k].y;
            float3 nn = float3(0.0, cos(inc), -sin(inc));
            float den = dot(rd, nn);
            if (abs(den) < 1e-4) continue;
            float t = -dot(ro, nn) / den;
            if (t <= 0.0 || t >= min(tBest, depth)) continue;
            float rr = length(ro + rd * t);
            float w = t * misc2.z * 0.6 / max(abs(den), 0.12);
            float cov = 1.0 - smoothstep(0.0, w, abs(rr - R));
            col = lerp(col, lineCol, cov * misc.x);
        }
    }
    {   // comet coma
        float3 oc = cometP.xyz - ro;
        float tca = dot(oc, rd);
        if (tca > 0.0 && tca < tBest + 3.0 * bPosR[B_COMET].w)
        {
            float d2 = dot(oc, oc) - tca * tca;
            float sg = max(0.22, tca * misc2.z * 2.2);
            col += float3(0.6, 0.8, 1.0) * exp(-d2 / (sg * sg)) * (0.10 + 0.50 * cometP.w);
        }
    }
    depthOut = min(depth, tBest);
    return col;
}

float3 PixelRay(float2 px)
{
    float u = (px.x / float(dims.x) * 2.0 - 1.0) * camRight.w;
    float v = (1.0 - px.y / float(dims.y) * 2.0) * camUp.w;
    return normalize(camFwd.xyz + camRight.xyz * u + camUp.xyz * v);
}

[numthreads(8, 8, 1)]
void CS_Render(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    int S = dims.z;
    float3 acc = 0.0;
    float dmin = 1e30;
    [loop] for (int sy = 0; sy < S; sy++)
        [loop] for (int sx = 0; sx < S; sx++)
        {
            float2 px = float2(id.xy) + (float2(sx, sy) + 0.5) / float(S);
            float d;
            float3 c = Trace(camPos.xyz, PixelRay(px), d);
            acc += min(c, 30.0);
            dmin = min(dmin, d);
        }
    uHdr[id.xy] = float4(acc / float(S * S), 1.0);
    uDepth[id.xy] = dmin;
}

// ------------------------------------------------------------------ scatter: 3D asteroids and comet tail particles
float Rnd1(uint i, uint salt)
{
    uint h = i * 747796405u + salt * 2891336453u + 12345u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return float(h & 0xFFFFFFu) * (1.0 / 16777216.0);
}

[numthreads(64, 1, 1)]
void CS_Clear(uint3 id : SV_DispatchThreadID)
{
    uint n = (uint)dims.x * (uint)dims.y * 3u;
    if (id.x < n) uScatter[id.x] = 0u;
}

[numthreads(64, 1, 1)]
void CS_Scatter(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    uint nA = (uint)misc2.x, nT = (uint)misc2.y;
    if (i >= nA + nT) return;
    float3 p, col;
    float simT = camFwd.w;
    if (i < nA)
    {
        float R = 21.0 + 5.0 * Rnd1(i, 1u);
        float ph = TAU * Rnd1(i, 2u);
        float y = (Rnd1(i, 3u) + Rnd1(i, 4u) + Rnd1(i, 5u) - 1.5) * 0.9;
        float b = 0.10 + 0.24 * Rnd1(i, 6u);
        float th = ph + TAU * simT / pow(R / 13.5, 1.5);
        p = float3(R * cos(th), y, R * sin(th));
        col = float3(0.85, 0.78, 0.68) * b * SunIntensity(R) * 0.30;
    }
    else
    {
        uint j = i - nA;
        float a = Rnd1(j, 11u), bj = Rnd1(j, 12u) * 2.0 - 1.0, cj = Rnd1(j, 13u) * 2.0 - 1.0;
        float L = cometV.w;
        float s = pow(a, 1.6) * L;
        float3 head = cometP.xyz;
        float3 away = normalize(head);
        float3 u1 = normalize(cross(away, float3(0, 1, 0)));
        float3 u2 = cross(away, u1);
        float spread = 0.04 + 0.07 * s;
        p = head + away * s + u1 * (bj * spread) + u2 * (cj * spread);
        bool dust = (j % 3u) == 0u;
        if (dust) p -= cometV.xyz * (s * s * 0.025);
        float fade = pow(saturate(1.0 - s / L), 1.3) * cometP.w;
        col = (dust ? float3(1.0, 0.88, 0.70) : float3(0.45, 0.72, 1.0)) * 0.20 * fade;
    }
    float3 v = p - camPos.xyz;
    float z = dot(v, camFwd.xyz);
    if (z < 0.2) return;
    float dist = length(v);
    if (i >= nA) col *= smoothstep(3.0, 14.0, dist) * (dist > 12.0 ? 12.0 / dist : 1.0);
    float x = dot(v, camRight.xyz) / z / camRight.w;
    float y2 = dot(v, camUp.xyz) / z / camUp.w;
    float sx = (x * 0.5 + 0.5) * float(dims.x) - 0.5;
    float sy = (0.5 - y2 * 0.5) * float(dims.y) - 0.5;
    int ix = (int)floor(sx), iy = (int)floor(sy);
    float fx = sx - float(ix), fy = sy - float(iy);
    [unroll] for (int k = 0; k < 4; k++)
    {
        int px = ix + (k & 1), py = iy + (k >> 1);
        if (px < 0 || py < 0 || px >= dims.x || py >= dims.y) continue;
        if (dist > tDepth[int2(px, py)]) continue;
        float w = ((k & 1) ? fx : 1.0 - fx) * ((k >> 1) ? fy : 1.0 - fy);
        uint idx = (uint)(py * dims.x + px) * 3u;
        InterlockedAdd(uScatter[idx + 0u], (uint)(col.x * w * 4096.0));
        InterlockedAdd(uScatter[idx + 1u], (uint)(col.y * w * 4096.0));
        InterlockedAdd(uScatter[idx + 2u], (uint)(col.z * w * 4096.0));
    }
}

float3 ScatterAt(int2 p)
{
    uint idx = (uint)(p.y * dims.x + p.x) * 3u;
    return float3(tScatter[idx], tScatter[idx + 1u], tScatter[idx + 2u]) * (1.0 / 4096.0);
}

// ------------------------------------------------------------------ bloom
float3 BrightOf(float3 c)
{
    float l = dot(c, float3(0.30, 0.59, 0.11));
    return c * saturate((l - 0.9) / 0.8);
}

float3 SceneAt(int2 p)
{
    p = clamp(p, int2(0, 0), int2(dims.x - 1, dims.y - 1));
    return tHdr[p].rgb + ScatterAt(p);
}

[numthreads(8, 8, 1)]
void CS_BrightH1(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    float3 acc = 0.0; float wsum = 0.0;
    [loop] for (int k = -10; k <= 10; k++)
    {
        float w = exp(-float(k * k) / 32.0);
        acc += BrightOf(SceneAt(int2(id.xy) + int2(k, 0))) * w;
        wsum += w;
    }
    uBlurOut[id.xy] = float4(acc / wsum, 1.0);
}

float3 BlurTap(int2 p, int2 dir, int stepPx)
{
    float3 acc = 0.0; float wsum = 0.0;
    [loop] for (int k = -10; k <= 10; k++)
    {
        float w = exp(-float(k * k) / 32.0);
        int2 q = clamp(p + dir * (k * stepPx), int2(0, 0), int2(dims.x - 1, dims.y - 1));
        acc += tBlurIn[q].rgb * w;
        wsum += w;
    }
    return acc / wsum;
}

[numthreads(8, 8, 1)]
void CS_BlurV1(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    uBlurOut[id.xy] = float4(BlurTap(int2(id.xy), int2(0, 1), 1), 1.0);
}

[numthreads(8, 8, 1)]
void CS_BlurH2(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    uBlurOut[id.xy] = float4(BlurTap(int2(id.xy), int2(1, 0), 4), 1.0);
}

[numthreads(8, 8, 1)]
void CS_BlurV2(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)dims.x || id.y >= (uint)dims.y) return;
    uBlurOut[id.xy] = float4(BlurTap(int2(id.xy), int2(0, 1), 4), 1.0);
}

float Tone(float c) { c = max(c, 0.0); return c < 0.8 ? c : 0.8 + 0.2 * (1.0 - exp(-(c - 0.8) * 5.0)); }

float3 SceneAtI(int2 p)
{
    p = clamp(p, int2(0, 0), int2(dims.x - 1, dims.y - 1));
    return tHdr[p].rgb + ScatterAt(p);
}

float3 BilScene(float2 f)
{
    float2 q = f - 0.5;
    float2 p0 = floor(q);
    float2 t = q - p0;
    int2 i0 = int2(p0);
    return lerp(lerp(SceneAtI(i0), SceneAtI(i0 + int2(1, 0)), t.x), lerp(SceneAtI(i0 + int2(0, 1)), SceneAtI(i0 + int2(1, 1)), t.x), t.y);
}

float3 BilTex(Texture2D<float4> tex, float2 f)
{
    float2 q = f - 0.5;
    float2 p0 = floor(q);
    float2 t = q - p0;
    int2 i0 = int2(p0);
    int2 mx = int2(dims.x - 1, dims.y - 1);
    float3 a = tex[clamp(i0, int2(0, 0), mx)].rgb;
    float3 b = tex[clamp(i0 + int2(1, 0), int2(0, 0), mx)].rgb;
    float3 c = tex[clamp(i0 + int2(0, 1), int2(0, 0), mx)].rgb;
    float3 d = tex[clamp(i0 + int2(1, 1), int2(0, 0), mx)].rgb;
    return lerp(lerp(a, b, t.x), lerp(c, d, t.x), t.y);
}

float3 LensFlare(float2 ndc, float aspect)
{
    float3 v = bPosR[B_SUN].xyz - camPos.xyz;
    float z = dot(v, camFwd.xyz);
    if (z <= 0.5) return 0.0;
    float2 sn = float2(dot(v, camRight.xyz) / z / camRight.w, dot(v, camUp.xyz) / z / camUp.w);
    float edge = max(abs(sn.x), abs(sn.y));
    float fade = saturate(1.6 - edge);
    if (fade <= 0.0) return 0.0;
    float vis = 1.0;
    float2 ip = float2((sn.x * 0.5 + 0.5) * float(dims.x), (0.5 - sn.y * 0.5) * float(dims.y));
    if (ip.x >= 0.0 && ip.y >= 0.0 && ip.x < float(dims.x) && ip.y < float(dims.y))
        vis = tDepth[int2(ip)] > length(v) * 0.92 ? 1.0 : 0.12;
    float2 d0 = (ndc - sn) * float2(aspect, 1.0);
    float3 acc = float3(1.0, 0.80, 0.55) * exp(-abs(d0.y) * 34.0) * exp(-abs(d0.x) * 1.3) * 0.05;
    [unroll] for (int k = 0; k < 4; k++)
    {
        float s = k == 0 ? 0.35 : (k == 1 ? 0.75 : (k == 2 ? 1.35 : 2.1));
        float rg = k == 0 ? 0.12 : (k == 1 ? 0.07 : (k == 2 ? 0.20 : 0.10));
        float2 dg = (ndc + sn * s) * float2(aspect, 1.0);
        float r = (length(dg) - rg) / (rg * 0.35);
        float ring = exp(-r * r);
        float disc = exp(-dot(dg, dg) / (rg * rg * 0.5));
        float3 tint = k == 0 ? float3(0.4, 0.7, 1.0) : (k == 1 ? float3(1.0, 0.5, 0.3) : (k == 2 ? float3(0.4, 1.0, 0.6) : float3(0.8, 0.5, 1.0)));
        acc += tint * (ring * 0.04 + disc * 0.018);
    }
    return acc * vis * fade;
}

[numthreads(8, 8, 1)]
void CS_Compose(uint3 id : SV_DispatchThreadID)
{
    int ow = dims3.x, oh = dims3.y;
    if (id.x >= (uint)ow || id.y >= (uint)oh) return;
    int2 p = int2(id.xy);
    float3 c;
    if (ow == dims.x && oh == dims.y)
        c = tHdr[p].rgb + ScatterAt(p) + tBlurB[p].rgb * misc.y + tBlurC[p].rgb * misc.z;
    else
    {
        float2 f = (float2(p) + 0.5) * float2(float(dims.x), float(dims.y)) / float2(float(ow), float(oh));
        c = BilScene(f) + BilTex(tBlurB, f) * misc.y + BilTex(tBlurC, f) * misc.z;
    }
    float2 ndc = float2((float(p.x) + 0.5) / float(ow) * 2.0 - 1.0, 1.0 - (float(p.y) + 0.5) / float(oh) * 2.0);
    if (misc3.x > 0.0) c += LensFlare(ndc, float(ow) / float(oh)) * misc3.x;
    c *= misc.w;
    c *= 1.0 - 0.18 * dot(ndc, ndc);                     // gentle vignette
    float3 o = float3(Tone(c.x), Tone(c.y), Tone(c.z));
    o = sqrt(o);                                         // display gamma
    if (dims3.z != 0)
    {
        float4 ov = tOverlay[p];
        o = lerp(o, 0.0, ov.g * 0.75);
        o = lerp(o, float3(0.84, 0.87, 0.94), ov.r);
    }
    o += (Hash3(int3(p, 7)) - 0.5) * (1.0 / 255.0);      // dither against banding
    uPix[p] = float4(saturate(o), 1.0);
}

// ------------------------------------------------------------------ best two-color fit per terminal cell
[numthreads(8, 8, 1)]
void CS_Cells(uint3 id : SV_DispatchThreadID)
{
    int cx = (int)id.x, cy = (int)id.y;
    int cellsX = dims.w, cellsY = dims2.x;
    if (cx >= cellsX || cy >= cellsY) return;
    int sx = dims2.y, sy = dims2.z;
    int n = sx * sy;
    int3 px[6];
    [unroll] for (int z = 0; z < 6; z++) px[z] = int3(0, 0, 0);
    [loop] for (int j = 0; j < sy; j++)
        [loop] for (int i = 0; i < sx; i++)
            px[j * sx + i] = (int3)(saturate(tPix[int2(cx * sx + i, cy * sy + j)].rgb) * 255.0 + 0.5);

    int3 mn = int3(255, 255, 255), mx = int3(0, 0, 0), sumAll = int3(0, 0, 0);
    [loop] for (int k = 0; k < n; k++) { mn = min(mn, px[k]); mx = max(mx, px[k]); sumAll += px[k]; }
    int3 fgm, bgm;
    int best = 0;
    int3 spread = mx - mn;
    if (spread.x < 6 && spread.y < 6 && spread.z < 6)
    {
        fgm = sumAll / n; bgm = fgm;
    }
    else
    {
        int bestErr = 0x7fffffff;
        int nm = 1 << (n - 1);
        fgm = sumAll / n; bgm = fgm;
        [loop] for (int m = 1; m < nm; m++)
        {
            int3 sa = int3(0, 0, 0), sb = int3(0, 0, 0);
            int ca = 0, cb = 0;
            [loop] for (int q = 0; q < n; q++)
            {
                if ((m >> q) & 1) { sa += px[q]; ca++; } else { sb += px[q]; cb++; }
            }
            int3 ma = sa / max(ca, 1), mb = sb / max(cb, 1);
            int err = 0;
            [loop] for (int r = 0; r < n; r++)
            {
                int3 d = px[r] - (((m >> r) & 1) ? ma : mb);
                err += d.x * d.x + d.y * d.y + d.z * d.z;
            }
            if (err < bestErr) { bestErr = err; best = m; fgm = ma; bgm = mb; }
        }
        int3 mean = sumAll / n;
        int flatErr = 0;
        [loop] for (int u = 0; u < n; u++) { int3 d = px[u] - mean; flatErr += d.x * d.x + d.y * d.y + d.z * d.z; }
        if (flatErr <= bestErr + 24) { best = 0; fgm = mean; bgm = mean; }
    }
    fgm &= 0xFC; bgm &= 0xFC;
    uint fg = ((uint)fgm.x << 16) | ((uint)fgm.y << 8) | (uint)fgm.z;
    uint bg = ((uint)bgm.x << 16) | ((uint)bgm.y << 8) | (uint)bgm.z;
    uCells[cy * cellsX + cx] = uint2(fg | ((uint)best << 24), bg);
}

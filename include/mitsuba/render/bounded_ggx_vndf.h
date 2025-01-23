#pragma once
#define BOUNDED_GGX_VNDF_H

#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/tensor.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class BoundedGGX {
public:
    MI_IMPORT_TYPES();

    explicit BoundedGGX(const float alpha, const float epsilon = 1e-3)
        : m_alpha(alpha), m_epsilon(epsilon), m_alpha2(alpha * alpha),
          m_alpha4(alpha * alpha * alpha * alpha), m_alpha_inv(1.f / alpha),
          m_alpha_inv2(1.f / (alpha * alpha)) {}

    Normal3f sample(const Vector3f &wi,
                    const Float &sample_phi,
                    const Float &sample_theta) const {
        Vector3f i_std = dr::normalize(
            Vector3f(wi.x() * this->m_alpha, wi.y() * this->m_alpha, wi.z()));

        Float u1 = this->clip_uniform(sample_phi);
        Float u2 = this->clip_uniform(sample_theta);

        Float phi = 2.f * dr::Pi<Float> * u1;
        Float s   = 1.f + dr::sqrt(wi.x() * wi.x() + wi.y() * wi.y());

        const auto &a2 = this->m_alpha2;

        Float s2 = s * s;
        Float k  = (1.f - a2) * s2 / (s2 + a2 * wi.z() * wi.z());

        Float lower_bound = -k * i_std.z();
        Float z = dr::fmadd(lower_bound, sample_theta, 1.f - sample_theta);

        Float sin_theta = dr::sqrt(dr::clip(1 - z * z, 0, 1));
        Vector3f o_std =
            Vector3f(sin_theta * dr::cos(phi), sin_theta * dr::sin(phi), z);
        Vector3f m_std = i_std + o_std;

        return dr::normalize(Normal3f(m_std.x() * this->m_alpha,
                                      m_std.y() * this->m_alpha, m_std.z()));
    }

    Float kiz_root(const Vector3f &wi) const {
        Float s        = 1 + dr::sqrt(wi.x() * wi.x() + wi.y() * wi.y());
        const auto &a2 = this->m_alpha2;
        Float s2       = s * s;
        Float k        = (1.f - a2) * s2 / (s2 + a2 * wi.z() * wi.z());
        const auto wi2 = wi * wi;
        return k * wi.z() + dr::sqrt(a2 * wi2.x() + a2 * wi2.y() + wi2.z());
    }

    Float pdf(const Vector3f &wi, const Vector3f &wo) const {
        Normal3f m  = dr::normalize(wi + wo);
        Float ndf   = this->ndf_supplementary(m);
        Vector2f ai = this->m_alpha * Vector2f(wi.x(), wi.y());
        Float len2  = dr::dot(ai, ai);
        Float t     = dr::sqrt(len2 + wi.z() * wi.z());

        Float s        = 1 + dr::sqrt(wi.x() * wi.x() + wi.y() * wi.y());
        const auto &a2 = this->m_alpha2;
        Float s2       = s * s;
        Float k        = (1.f - a2) * s2 / (s2 + a2 * wi.z() * wi.z());

        return ndf / (2 * (k * wi.z() + t));
        // return dr::select(wi.z() >= 0, ndf / (2 * (k * wi.z() + t)),
        // ndf * (t - wi.z()) / (2 * len2));
    }

    Vector2f invert(const Vector3f &wi, const Vector3f &m) const {
        const auto &a2 = this->m_alpha2;
        Normal3f i_std = dr::normalize(
            Normal3f(wi.x() * this->m_alpha, wi.y() * this->m_alpha, wi.z()));

        Float Nx = m.x(), Ny = m.y(), Nz = m.z();
        Float Ix = i_std.x(), Iy = i_std.y(), Iz = i_std.z();

        Float denom = (Nx * Nx + Ny * Ny) * m_alpha_inv2 + Nz * Nz;
        Float numer = 2.0 * ((Ix * Nx + Iy * Ny) * m_alpha_inv + Iz * Nz);
        Float lam   = numer / denom;

        Float ox = lam * Nx * m_alpha_inv - Ix;
        Float oy = lam * Ny * m_alpha_inv - Iy;
        Float oz = lam * Nz - Iz;

        Float phi = dr::atan2(oy, ox);
        phi       = dr::select(phi < 0.f, phi + 2.f * dr::Pi<Float>, phi);

        Float z  = oz;
        Float s  = 1.f + dr::sqrt(wi.x() * wi.x() + wi.y() * wi.y());
        Float s2 = s * s;
        Float k  = (1.f - a2) * s2 / (s2 + a2 * wi.z() * wi.z());
        Float lower_bound =
            dr::select(wi.z() > 0.f, -k * i_std.z(), -i_std.z());

        Float u2 = (z - 1.0) / (lower_bound - 1.0);
        Float u1 = phi / (2 * dr::Pi<Float>);

        u1 = clip_uniform(u1);
        u2 = clip_uniform(u2);

        return Vector2f(u1, u2);
    }

    Float lambda(const Float &theta) const {
        Float tan2Theta       = dr::square(dr::tan(theta));
        const Float nominator = dr::sqrt(1 + this->m_alpha2 * tan2Theta) - 1;
        return dr::select(dr::isinf(tan2Theta), 0.f, nominator / 2.f);
    }

    Float sigma(const Float &theta) const {
        return dr::cos(theta) * (1.f + this->lambda(theta));
    }

    Float
    smith_g(const Vector3f &wi, const Vector3f &wo, const Vector3f &m) const {
        return this->smith_g1(wi, m) * this->smith_g1(wo, m);
    }

    /**
     * Numerically stable method computing the elevation of the given
     * (normalized) vector in the local frame.
     * Conceptually equivalent to:
     *     safe_acos(Frame3f::cos_theta(d))
     */
    auto elevation(const Vector3f &d) const {
        auto dist = dr::sqrt(dr::square(d.x()) + dr::square(d.y()) +
                             dr::square(d.z() - 1.f));
        return 2.f * dr::safe_asin(.5f * dist);
    }

    Float smith_g1(const Vector3f &wo, const Vector3f &wm) const {
        return dr::select(dr::dot(wo, wm) > 0.f,
                          1.f / (1.f + this->lambda(this->elevation(wo))), 0.f);
    }

    Float sigma_inv(const Float &sigma) const {
        const auto &a2 = this->m_alpha2;
        const auto &a4 = this->m_alpha4;
        const auto s2  = sigma * sigma;
        const auto nominator =
            2.f * sigma - dr::sqrt(a4 + 4 * s2 - 4 * a2 * s2);
        return dr::acos(nominator / 2.f);
    }

    Float ndf(const Vector3f &n) const {
        Vector3f n2       = n * n;
        const auto &a2    = this->m_alpha2;
        Float denominator = n2.x() / a2 + n2.y() / a2 + n2.z();
        denominator       = dr::Pi<Float> * a2 * denominator * denominator;
        return 1.f / denominator;
    }

    Float ndf_supplementary(const Vector3f &m) const {
        const auto &a2         = this->m_alpha2;
        const auto mx          = dr::square(m.x()) / a2;
        const auto my          = dr::square(m.y()) / a2;
        const auto mz          = dr::square(m.z());
        const auto denominator = dr::Pi<Float> * a2 * dr::square(mx + my + mz);
        return dr::select(m.z() > 0, 1.f / denominator, 0);
    }

    [[nodiscard]] float alpha() const { return this->m_alpha; }

    [[nodiscard]] float epsilon() const { return this->m_epsilon; }

    Float theta2u(Float theta) const {
        return dr::sqrt(theta * (2.f / dr::Pi<Float>) );
    }

    Float s(Float ix, Float iy) const {
        return 1 + dr::sqrt(ix * ix + iy * iy);
    }

    Float k(Float s, Float z_i) const {
        const auto a2 = this->m_alpha2;
        const auto s2 = s * s;
        return (1.f - a2) * s2 / (s2 + a2 * z_i * z_i);
    }

    Float z_i_std(Float ix, Float iy, Float iz) const {
        const auto ax  = this->m_alpha * ix;
        const auto ay  = this->m_alpha * iy;
        const auto len = dr::sqrt(ax * ax + ay * ay + iz * iz);
        return iz / len;
    }

    Float lower_bound(Float k, Float z_i_std) const { return -k * z_i_std; }

    Float z(Float u, Float lb) const { return dr::fmadd(lb, u, 1.f - lb); }

    // Float theta_jacobian(Float u, const Vector3f &wi) const {
    //     const auto ix = wi.x(), iy = wi.y(), iz = wi.z();
    //     const auto k_val = k(s(ix, iy), z_i_std(ix, iy, iz));
    //
    //     const auto nominator = iz * k_val + 1;
    //     const auto denominator =
    //         dr::sqrt(-dr::square(u * iz * k_val + u - 1) + 1);
    //
    //     return nominator / denominator;
    // }

    Float theta_jacobian(const Float &u, const Vector3f &wi) const {
        // Variables you need to define/assign elsewhere
        Vector3f i_std = dr::normalize(
            Vector3f(wi.x() * this->m_alpha, wi.y() * this->m_alpha, wi.z()));
        Float s = this->s(wi.x(), wi.y());
        Float k = this->k(s, wi.z());

        Float lambda_ = lower_bound(k, i_std.z()); // e.g. lambda_ = ...
        Float alpha   = this->m_alpha;             // e.g. alpha  = ...
        Float i_std_x = i_std.x();                 // e.g. i_std_x = ...
        Float i_std_y = i_std.y();                 // e.g. i_std_y = ...

        // Example pi definition (if not already available in your code)
        const Float pi = dr::Pi<Float>;

        // 1) Common subexpressions
        //    A = lambda_*u^2 - u^2 + 1 = (lambda_ - 1)*u^2 + 1
        Float u2 = u * u;
        Float A  = (lambda_ - Float(1)) * u2 + Float(1);

        // 2) B = sqrt(1 - A^2)
        Float A2 = A * A;
        Float B  = sqrt(Float(1) - A2);

        // 3) C = 2 * pi * u^2
        Float C    = Float(2) * pi * u2;
        Float cosC = cos(C);
        Float sinC = sin(C);

        // 4) D = lambda_*u - u = u*(lambda_ - 1)
        Float D = u * (lambda_ - Float(1));

        // 5) E = B*cosC + i_std_x
        //    F = B*sinC + i_std_y
        Float E = B * cosC + i_std_x;
        Float F = B * sinC + i_std_y;

        // 6) G = A + i_std_x  (since A = (lambda_*u^2 - u^2 + 1))
        Float G = A + i_std_x;

        // 7) H = (E^2 + F^2)*alpha^2 + G^2
        //    then we need H^(3/2) and sqrt(H)
        Float E2     = E * E;
        Float F2     = F * F;
        Float G2     = G * G;
        Float alpha2 = alpha * alpha;

        Float H     = (E2 + F2) * alpha2 + G2;
        Float sqrtH = sqrt(H);
        Float H32   = H * sqrtH; // H^(3/2)

        // 8) Intermediate terms matching your expression
        //    term1 = 2*pi*B*u*sinC + (A*D*cosC)/B
        //    term2 = 2*pi*B*u*cosC - (A*D*sinC)/B
        Float term1 = Float(2) * pi * B * u * sinC + (A * D * cosC) / B;
        Float term2 = Float(2) * pi * B * u * cosC - (A * D * sinC) / B;

        // 9) Combine them according to the expression
        //    bracket = term1*E*alpha^2 - term2*F*alpha^2 - G*D
        Float bracket = term1 * E * alpha2 - term2 * F * alpha2 - G * D;

        // 10) The big fraction part:
        //     2 * ((bracket) * G) / H^(3/2)
        Float fractionPart = Float(2) * (bracket * G) / H32;

        // 11) The additional term:
        //     2 * D / sqrt(H)
        Float additionalTerm = Float(2) * D / sqrtH;

        // 12) Final result = fractionPart + additionalTerm
        Float result = fractionPart + additionalTerm;

        return result;
    }

    friend std::ostream &operator<<(std::ostream &os, const BoundedGGX &ggx) {
        os << "BoundedGGX[\n";
        os << "\talpha=" << ggx.m_alpha << "\n";
        os << "\tepsilon=" << ggx.m_epsilon << "\n";
        os << "]" << std::endl;
        return os;
    }

    Float u_to_param(Float u) const {
        return dr::square(u);
    }

    Float param_to_u(Float p) const {
        return dr::sqrt(p);
    }

    Float param_jacobian(Float p) const {
        return 1 / (2.f * p);
    }

private:
    Float clip_uniform(const Float &u) const {
        return u;
        // return dr::clip(u, this->m_epsilon, 1.0 - this->m_epsilon);
    }

    float m_alpha;
    float m_epsilon;
    float m_alpha2;
    float m_alpha4;
    float m_alpha_inv;
    float m_alpha_inv2;
};

NAMESPACE_END(mitsuba)

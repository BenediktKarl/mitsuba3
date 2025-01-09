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
        : m_alpha(alpha), m_epsilon(epsilon) {
        this->m_alpha2     = this->m_alpha * this->m_alpha;
        this->m_alpha4     = this->m_alpha2 * this->m_alpha2;
        this->m_alpha_inv  = 1.0 / this->m_alpha;
        this->m_alpha_inv2 = this->m_alpha_inv * this->m_alpha_inv;
    }

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

        Float lower_bound =
            dr::select(wi.z() > 0.f, -k * i_std.z(), -i_std.z());
        Float z = dr::fmadd(lower_bound, u2, 1.f - u2);

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

        Float denominator =
            (m.x() * m.x() + m.y() * m.y()) * m_alpha_inv2 + m.z();
        Float numerator =
            2.f * ((i_std.x() * m.x() + i_std.y() * m.y()) * m_alpha_inv +
                   i_std.z() * m.z());
        Float lam = numerator / denominator;

        Vector3f o =
            (lam * m.x() * m_alpha_inv - i_std.x(),
             lam * m.y() * m_alpha_inv - i_std.y(), lam * m.z() - i_std.z());

        Float phi = dr::atan2(o.y(), o.x());
        phi       = dr::select(phi < 0.f, phi + 2.f * dr::Pi<Float>, phi);

        Float z  = o.z();
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
        const Float a   = 1.f / (this->m_alpha * dr::tan(theta));
        Float nominator = -1.f + dr::sqrt(1.f + 1.f / (a * a));
        return nominator / 2.f;
    }

    Float sigma(const Float &theta) const {
        return dr::cos(theta) * (1.f + this->lambda(theta));
    }

    Float smith_g(const Vector3f &wi, const Vector3f &wo, const Vector3f &m) const {
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

private:
    Float clip_uniform(const Float &u) const {
        return dr::clip(u, this->m_epsilon, 1.0 - this->m_epsilon);
    }

    float m_alpha;
    float m_epsilon;
    float m_alpha2;
    float m_alpha4;
    float m_alpha_inv;
    float m_alpha_inv2;
};

NAMESPACE_END(mitsuba)

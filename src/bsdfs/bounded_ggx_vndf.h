//
// Created by Bene on 17/12/2024.
//

#ifndef BOUNDED_GGX_VNDF_H
#define BOUNDED_GGX_VNDF_H

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
                    const Float &sample1,
                    const Float &sample2) const {
        Vector3f i_std = dr::normalize(
            Vector3f(wi.x() * this->m_alpha, wi.y() * this->m_alpha, wi.z()));

        Float u1 = this->clip_uniform(sample1);
        Float u2 = this->clip_uniform(sample2);

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

    Float pdf(const Vector3f &wi, const Vector3f &wo) const {
        Normal3f m  = dr::normalize(wi + wo);
        Float ndf   = this->ggx_ndf(m);
        Vector2f ai = this->m_alpha * Vector2f(wi.x(), wi.y());
        Float len2  = dr::dot(ai, ai);
        Float t     = dr::sqrt(len2 + wi.z() * wi.z());

        Float s        = 1 + dr::sqrt(wi.x() * wi.x() + wi.y() * wi.y());
        const auto &a2 = this->m_alpha2;
        Float s2       = s * s;
        Float k        = (1.f - a2) * s2 / (s2 + a2 * wi.z() * wi.z());

        return dr::select(wi.z() >= 0, ndf / (2 * (k * wi.z() + t)),
                          ndf * (t - wi.z()) / (2 * len2));
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
        Float lower_bound = dr::select(wi.z() > 0.f, -k * i_std.z(), -i_std.z());

        Float u2 = (z - 1.0) / (lower_bound - 1.0);
        Float u1 = phi / (2 * dr::Pi<Float>);

        u1 = clip_uniform(u1);
        u2 = clip_uniform(u2);

        return Vector2f(u1, u2);
    }

private:
    Float clip_uniform(const Float &u) const {
        return dr::clip(u, this->m_epsilon, 1.0 - this->m_epsilon);
    }

    Float ggx_ndf(const Vector3f &n) const {
        Vector3f n2       = n * n;
        const auto &a2    = this->m_alpha2;
        Float denominator = n2.x() / a2 + n2.y() / a2 + n2.z();
        denominator       = dr::Pi<Float> * a2 * denominator * denominator;
        return 1.f / denominator;
    }

    float m_alpha;
    float m_epsilon;
    float m_alpha2;
    float m_alpha4;
    float m_alpha_inv;
    float m_alpha_inv2;
};

NAMESPACE_END(mitsuba)

#endif // BOUNDED_GGX_VNDF_H

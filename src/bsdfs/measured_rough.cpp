#include <mitsuba/core/properties.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/tensor.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/microfacet.h>
#include <array>
#include <cmath>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MeasuredRough final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES()

    using Warp2D0 = Marginal2D<Float, 0, true>;
    using Warp2D2 = Marginal2D<Float, 2, true>;
    using Warp2D3 = Marginal2D<Float, 3, true>;

    MeasuredRough(const Properties &props) : Base(props) {
        m_components.push_back(BSDFFlags::GlossyReflection | BSDFFlags::FrontSide);
        m_flags = m_components[0];

        MicrofacetDistribution<ScalarFloat, Spectrum> distr(props);
        m_type = distr.type();
        m_sample_visible = distr.sample_visible();

        if (distr.is_anisotropic())
            Throw("The 'measuredrough' plugin currently does not support "
                  "anisotropic microfacet distributions!");

        m_alpha = distr.alpha();

        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_name             = file_path.filename().string();

        ref<TensorFile> tf = new TensorFile(file_path);
        using Field = TensorFile::Field;

        const Field &theta_i       = tf->field("theta_i");
        const Field &phi_i         = tf->field("phi_i");
        const Field &description   = tf->field("description");

        Field spectra, wavelengths;
        bool is_spectral = tf->has_field("wavelengths");

        const ScalarFloat rgb_wavelengths[3] = { 0, 1, 2 };
        if (is_spectral) {
            spectra = tf->field("spectra");
            wavelengths = tf->field("wavelengths");
            if constexpr (!is_spectral_v<Spectrum>)
                Throw("Measurements in spectral format require the use of a spectral variant of Mitsuba!");
        } else {
            spectra = tf->field("rgb");
            if constexpr (!is_rgb_v<Spectrum>)
                Throw("Measurements in RGB format require the use of a RGB variant of Mitsuba!");

            wavelengths.shape.push_back(3);
            wavelengths.data = rgb_wavelengths;
        }

        if (!(description.shape.size() == 1 &&
              description.dtype == Struct::Type::UInt8 &&

              theta_i.shape.size() == 1 &&
              theta_i.dtype == Struct::Type::Float32 &&

              phi_i.shape.size() == 1 &&
              phi_i.dtype == Struct::Type::Float32 &&

              (!is_spectral || (
                  wavelengths.shape.size() == 1 &&
                  wavelengths.dtype == Struct::Type::Float32
              )) &&

              spectra.dtype == Struct::Type::Float32 &&
              spectra.shape.size() == 5 &&
              spectra.shape[0] == phi_i.shape[0] &&
              spectra.shape[1] == theta_i.shape[0] &&
              spectra.shape[2] == (is_spectral ? wavelengths.shape[0] : 3) &&
              spectra.shape[3] == spectra.shape[4] &&

              luminance.shape[2] == spectra.shape[3] &&
              luminance.shape[3] == spectra.shape[4]))
              Throw("Invalid file structure: %s", tf);

        m_isotropic = phi_i.shape[0] <= 2;
        if (!m_isotropic) {
            Throw("The measuredrough plugin does currently not support anistropic materials");
        }

        if (!m_isotropic) {
            ScalarFloat *phi_i_data = (ScalarFloat *) phi_i.data;
            m_reduction = (int) std::rint((2 * dr::Pi<ScalarFloat>) /
                (phi_i_data[phi_i.shape[0] - 1] - phi_i_data[0]));
        }

        // Construct spectral interpolant
        m_spectra = Warp2D3(
            (ScalarFloat *) spectra.data,
            ScalarVector2u(spectra.shape[4], spectra.shape[3]),
            {{ (uint32_t) phi_i.shape[0],
               (uint32_t) theta_i.shape[0],
               (uint32_t) wavelengths.shape[0] }},
            {{ (const ScalarFloat *) phi_i.data,
               (const ScalarFloat *) theta_i.data,
               (const ScalarFloat *) wavelengths.data }},
            false, false
        );

        std::string description_str(
            (const char *) description.data,
            (const char *) description.data + description.shape[0]
        );

        Log(Info, "Loaded material \"%s\" (resolution %i x %i x %i x %i x %i)",
            description_str, spectra.shape[0], spectra.shape[1],
            spectra.shape[3], spectra.shape[4], spectra.shape[2]);
    }

    /**
     * Numerically stable method computing the elevation of the given
     * (normalized) vector in the local frame.
     * Conceptually equivalent to:
     *     safe_acos(Frame3f::cos_theta(d))
     */
    auto elevation(const Vector3f &d) const {
        auto dist = dr::sqrt(dr::square(d.x()) + dr::square(d.y()) + dr::square(d.z() - 1.f));
        return 2.f * dr::safe_asin(.5f * dist);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /*sample1*/,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Vector3f wi = si.wi;
        active &= Frame3f::cos_theta(wi) > 0;

        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active))
            return { bs, 0.f };

        Float sx = -1.f, sy = -1.f;

        if (m_reduction >= 2) {
            sy = wi.y();
            sx = (m_reduction == 4) ? wi.x() : sy;
            wi.x() = dr::mulsign_neg(wi.x(), sx);
            wi.y() = dr::mulsign_neg(wi.y(), sy);
        }

        Float theta_i = elevation(wi),
            phi_i   = dr::atan2(wi.y(), wi.x());

        Float params[2] = { phi_i, theta_i };
        Vector2f u_wi(theta2u(theta_i), phi2u(phi_i));

        Vector2f sample;

        sample = Vector2f(sample2.y(), sample2.x());
        Float pdf = 1.f;

        auto [u_m, ndf_pdf] = m_vndf.sample(sample, params, active);

        Float phi_m   = u2phi(u_m.y()),
              theta_m = u2theta(u_m.x());

        if (m_isotropic)
            phi_m += phi_i;

        // Spherical -> Cartesian coordinates
        auto [sin_phi_m, cos_phi_m] = dr::sincos(phi_m);
        auto [sin_theta_m, cos_theta_m] = dr::sincos(theta_m);

        Vector3f m(
            cos_phi_m * sin_theta_m,
            sin_phi_m * sin_theta_m,
            cos_theta_m
        );

        Float jacobian = dr::maximum(2.f * dr::square(dr::Pi<Float>) * u_m.x() *
                                    sin_theta_m, 1e-6f) * 4.f * dr::dot(wi, m);

        bs.wo = dr::fmsub(m, 2.f * dr::dot(m, wi), wi);
        bs.pdf = ndf_pdf * pdf / jacobian;

        bs.eta               = 1.f;
        bs.sampled_type      = +BSDFFlags::GlossyReflection;
        bs.sampled_component = 0;

        UnpolarizedSpectrum spec;
        for (size_t i = 0; i < dr::size_v<UnpolarizedSpectrum>; ++i) {
            Float params_spec[3] = { phi_i, theta_i,
                is_spectral_v<Spectrum> ? si.wavelengths[i] : Float((float) i) };
            spec[i] = m_spectra.eval(sample, params_spec, active);
        }

        if (m_jacobian)
            spec *= m_ndf.eval(u_m, params, active) /
                    (4 * m_sigma.eval(u_wi, params, active));

        bs.wo.x() = dr::mulsign_neg(bs.wo.x(), sx);
        bs.wo.y() = dr::mulsign_neg(bs.wo.y(), sy);

        active &= Frame3f::cos_theta(bs.wo) > 0;

        return { bs, (depolarizer<Spectrum>(spec) / bs.pdf) & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo_, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Vector3f wi = si.wi, wo = wo_;

        Float cos_theta_i = Frame3f::cos_theta(wi);
        active &= cos_theta_i > 0.f &&
                  Frame3f::cos_theta(wo) > 0.f;

        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active))
            return Spectrum(0.f);

        if (m_reduction >= 2) {
            Float sy = wi.y(),
                sx = (m_reduction == 4) ? wi.x() : sy;

            wi.x() = dr::mulsign_neg(wi.x(), sx);
            wi.y() = dr::mulsign_neg(wi.y(), sy);
            wo.x() = dr::mulsign_neg(wo.x(), sx);
            wo.y() = dr::mulsign_neg(wo.y(), sy);
        }

        UnpolarizedSpectrum spec;

        Float theta_i = elevation(wi);
        for (size_t i = 0; i < dr::size_v<UnpolarizedSpectrum>; ++i) {
            Float params_spec[3] = { 0.f, theta_i,
                is_spectral_v<Spectrum> ? si.wavelengths[i] : Float((float) i) };
            spec[i] = m_spectra.eval(sample, params_spec, active);
        }

        MicrofacetDistribution distr(m_type, m_alpha, m_sample_visible);

        // Calculate the reflection half-vector
        Vector3f H = dr::normalize(wo + si.wi);

        // Evaluate the microfacet normal distribution
        Float D = distr.eval(H);

        // Fresnel term
        Float F = std::get<0>(fresnel(dr::dot(si.wi, H), Float(1.49))); // TODO look at fresnel eta?

        // Smith's shadow-masking function
        Float G = distr.G(si.wi, wo, H);

        // Calculate the specular reflection component
        spec *= F * D * G / (4.f * cos_theta_i);

        return depolarizer<Spectrum>(spec) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo_, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Vector3f wi = si.wi, wo = wo_;

        active &= Frame3f::cos_theta(wi) > 0.f &&
                  Frame3f::cos_theta(wo) > 0.f;

        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) || dr::none_or<false>(active))
            return 0.f;

        if (m_reduction >= 2) {
            Float sy = wi.y(),
                sx = (m_reduction == 4) ? wi.x() : sy;

            wi.x() = dr::mulsign_neg(wi.x(), sx);
            wi.y() = dr::mulsign_neg(wi.y(), sy);
            wo.x() = dr::mulsign_neg(wo.x(), sx);
            wo.y() = dr::mulsign_neg(wo.y(), sy);
        }

        Vector3f m = dr::normalize(wo + wi);

        // Cartesian -> spherical coordinates
        Float theta_i = elevation(wi),
              phi_i   = dr::atan2(wi.y(), wi.x()),
              theta_m = elevation(m),
              phi_m   = dr::atan2(m.y(), m.x());

        // Spherical coordinates -> unit coordinate system
        Vector2f u_wi(theta2u(theta_i), phi2u(phi_i));
        Vector2f u_m (theta2u(theta_m),
                      phi2u(m_isotropic ? (phi_m - phi_i) : phi_m));

        u_m[1] = u_m[1] - dr::floor(u_m[1]);

        Float params[2] = { phi_i, theta_i };
        auto [sample, vndf_pdf] = m_vndf.invert(u_m, params, active);

        Float pdf = 1.f;
        #if MI_SAMPLE_LUMINANCE == 1
        pdf = m_luminance.eval(sample, params, active);
        #endif

        Float jacobian =
            dr::maximum(2.f * dr::square(dr::Pi<Float>) * u_m.x() * Frame3f::sin_theta(m), 1e-6f) * 4.f *
            dr::dot(wi, m);

        pdf = vndf_pdf * pdf / jacobian;

        return dr::select(active, pdf, 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MeasuredRough[" << std::endl
            << "  filename = \"" << m_name << "\"," << std::endl
            << "  spectra = " << string::indent(m_spectra.to_string()) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    template <typename Value> Value u2theta(Value u) const {
        return dr::square(u) * (dr::Pi<Float> / 2.f);
    }

    template <typename Value> Value u2phi(Value u) const {
        return (2.f * u - 1.f) * dr::Pi<Float>;
    }

    template <typename Value> Value theta2u(Value theta) const {
        return dr::sqrt(theta * (2.f / dr::Pi<Float>));
    }

    template <typename Value> Value phi2u(Value phi) const {
        return (phi + dr::Pi<Float>) * dr::InvTwoPi<Float>;
    }

private:
    std::string m_name;
    Warp2D3 m_spectra;

    /* Microfacet specific member variables */
    MicrofacetType m_type;
    Float m_alpha;
    bool m_sample_visible;

    bool m_isotropic;
    bool m_jacobian;
    int m_reduction;
};

MI_IMPLEMENT_CLASS_VARIANT(MeasuredRough, BSDF)
MI_EXPORT_PLUGIN(MeasuredRough, "Measured rough material")
NAMESPACE_END(mitsuba)

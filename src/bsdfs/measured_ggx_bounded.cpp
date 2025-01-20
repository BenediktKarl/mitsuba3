#include <array>
#include <cmath>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/tensor.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bounded_ggx_vndf.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

#define PHI_M_ISOTROPIC 1

template <typename Float, typename Spectrum>
class MeasuredGGXBounded final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, BoundedGGX)

    using Warp2D0 = Marginal2D<Float, 0, true>;
    using Warp2D2 = Marginal2D<Float, 2, true>;
    using Warp2D3 = Marginal2D<Float, 3, true>;

    MeasuredGGXBounded(const Properties &props) : Base(props) {
        m_components.push_back(BSDFFlags::GlossyReflection |
                               BSDFFlags::FrontSide);
        m_flags = m_components[0];

        if constexpr (!is_rgb_v<Spectrum>)
            Throw("Measurements in RGB format require the use of a RGB variant "
                  "of Mitsuba!");

        if (props.has_property("specular_reflectance")) {
            m_specular_reflectance =
                props.texture<Texture>("specular_reflectance", 1.f);
            m_alpha = props.get("alpha", 0.5f);
            Log(Info, "Loaded measured bounded GGX material in debug mode with "
                      "assigned texture color");
            return;
        }

        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));

        ref<TensorFile> tf = new TensorFile(file_path);
        using Field        = TensorFile::Field;

        const Field &theta_i = tf->field("theta_i");
        const Field &phi_i   = tf->field("phi_i");
        const Field &alpha   = tf->field("alpha");

        Field spectra, wavelengths;
        const ScalarFloat rgb_wavelengths[3] = { 0, 1, 2 };
        spectra                              = tf->field("rgb");

        wavelengths.shape.push_back(3);
        wavelengths.data = rgb_wavelengths;

        if (!(theta_i.shape.size() == 1 &&
              theta_i.dtype == Struct::Type::Float32 &&

              phi_i.shape.size() == 1 && phi_i.dtype == Struct::Type::Float32 &&

              spectra.dtype == Struct::Type::Float32 &&
              spectra.shape.size() == 5 && spectra.shape[0] == phi_i.shape[0] &&
              spectra.shape[1] == theta_i.shape[0] && spectra.shape[2] == 3 &&
              spectra.shape[3] == spectra.shape[4] &&

              alpha.shape.size() == 1 && alpha.shape[0] == 1))
            Throw("Invalid file structure: %s", tf);

        // Construct spectral interpolant
        m_spectra =
            Warp2D3((ScalarFloat *) spectra.data,
                    ScalarVector2u(spectra.shape[4], spectra.shape[3]),
                    { { (uint32_t) phi_i.shape[0], (uint32_t) theta_i.shape[0],
                        (uint32_t) wavelengths.shape[0] } },
                    { { (const ScalarFloat *) phi_i.data,
                        (const ScalarFloat *) theta_i.data,
                        (const ScalarFloat *) wavelengths.data } },
                    false, false);

        m_alpha =
            std::clamp(static_cast<const float *>(alpha.data)[0], 0.f, 1.f);

        Log(Info,
            "Loaded material measured bounded GGX material with roughness %f",
            this->m_alpha);
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

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /*sample1*/,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);
        auto bs        = dr::zeros<BSDFSample3f>();
        const auto &wi = si.wi;

        active &= Frame3f::cos_theta(wi) > 0;
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
            dr::none_or<false>(active))
            return { bs, 0.f };

        const BoundedGGX bounded_ggx(this->m_alpha);

        Float theta_i = elevation(wi), phi_i = dr::atan2(wi.y(), wi.x());
        Normal3f spectrum_m = bounded_ggx.sample(wi, sample2.y(), sample2.x());
        Normal3f reflection_m(spectrum_m);

#if PHI_M_ISOTROPIC == 1
        const auto theta_m = elevation(reflection_m);
        const auto phi_m   = dr::atan2(reflection_m.y(), reflection_m.x());
        const auto [sin_theta_m, cos_theta_m] = dr::sincos(theta_m);
        const auto [sin_phi_m, cos_phi_m]     = dr::sincos(phi_m + phi_i);
        reflection_m = Vector3f(cos_phi_m * sin_theta_m,
                                sin_phi_m * sin_theta_m, cos_theta_m);
#endif

        Vector3f wo = dr::fmsub(reflection_m, 2.f * dr::dot(reflection_m, wi), wi);

        bs.wo                = wo;
        bs.eta               = 1.f;
        bs.sampled_type      = +BSDFFlags::GlossyReflection;
        bs.sampled_component = 0;

        bs.wo.x() = dr::mulsign_neg(bs.wo.x(), -1.f);
        bs.wo.y() = dr::mulsign_neg(bs.wo.y(), -1.f);

        auto spec = this->eval_m(ctx, si, sample2, active);
        bs.pdf    = this->pdf(ctx, si, wo, active);

        spec /= bs.pdf;

        active &= Frame3f::cos_theta(bs.wo) > 0;
        active &= bs.pdf > 0;

        return { bs, (depolarizer<Spectrum>(spec)) & active };
    }

    Spectrum eval(const BSDFContext &ctx,
                  const SurfaceInteraction3f &si,
                  const Vector3f &wo_,
                  Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
            dr::none_or<false>(active)) {
            return Spectrum(0.f);
        }

        const Vector3f wi = si.wi, &wo = wo_;
        active &= Frame3f::cos_theta(wi) > 0.f && Frame3f::cos_theta(wo) > 0.f;

        Float phi_i = dr::atan2(wi.y(), wi.x());
        Float phi_o = dr::atan2(wo.y(), wi.x());
        Vector3f m  = dr::normalize(wo + wi);

#if PHI_M_ISOTROPIC == 1
        const auto theta_m                    = elevation(m);
        const auto phi_m                      = dr::atan2(m.y(), m.x());
        const auto [sin_theta_m, cos_theta_m] = dr::sincos(theta_m);
        const auto [sin_phi_m, cos_phi_m]     = dr::sincos(phi_m - phi_o);
        m = Vector3f(cos_phi_m * sin_theta_m, sin_phi_m * sin_theta_m,
                     cos_theta_m);
#endif

        const auto bounded_ggx = BoundedGGX(this->m_alpha);
        const auto sample2 = bounded_ggx.invert(wi, m);
        const auto sample  = Point2f(sample2.y(), sample2.x());

        auto spec = this->eval_m(ctx, si, sample, active);
        return spec;
    }

    Spectrum eval_m(const BSDFContext &ctx,
                    const SurfaceInteraction3f &si,
                    const Vector2f &sample,
                    Mask active) const {
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
            dr::none_or<false>(active)) {
            return Spectrum(0.f);
        }

        const Vector3f wi  = si.wi;

        Float theta_i = elevation(wi), phi_i = dr::atan2(wi.y(), wi.x());

        UnpolarizedSpectrum spec;
        for (size_t i = 0; i < dr::size_v<UnpolarizedSpectrum>; ++i) {
            Float params_spec[3] = { phi_i, theta_i,
                                     Float(static_cast<float>(i)) };
            if (!m_specular_reflectance) {
                spec[i] = this->m_spectra.eval(sample, params_spec, active);
            } else {
                spec[i] = 1;
            }
        }

        if (m_specular_reflectance) {
            spec *= m_specular_reflectance->eval(si, active);
        }

        // if (m_specular_reflectance) {
        // spec *= bounded_ggx.ndf_supplementary(m) * bounded_ggx.smith_g(wi,
        // wo, m) / (4.f * Frame3f::cos_theta(si.wi)); } else { spec *=
        // bounded_ggx.ndf_supplementary(m) / (4.f *
        // bounded_ggx.sigma(elevation(si.wi)));
        // }

        return depolarizer<Spectrum>(spec) & active;
    }

    Float pdf(const BSDFContext &ctx,
              const SurfaceInteraction3f &si,
              const Vector3f &wo_,
              Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        if (!ctx.is_enabled(BSDFFlags::GlossyReflection) ||
            dr::none_or<false>(active)) {
            return 0.f;
        }

        const BoundedGGX bounded_ggx(this->m_alpha);

        const Vector3f wi = si.wi, &wo = wo_;
        active &= Frame3f::cos_theta(wi) > 0.f && Frame3f::cos_theta(wo) > 0.f;

        const auto vndf_pdf = bounded_ggx.pdf(wi, wo);
        return dr::select(active, vndf_pdf, 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MeasuredBoundedGGX[" << std::endl
            << "  spectra = " << string::indent(m_spectra.to_string())
            << std::endl
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
        return dr::sqrt(theta * (2.f / dr::Pi<Float>) );
    }

    template <typename Value> Value phi2u(Value phi) const {
        return (phi + dr::Pi<Float>) *dr::InvTwoPi<Float>;
    }

    Warp2D3 m_spectra;
    float m_alpha;
    ref<Texture> m_specular_reflectance;
};

MI_IMPLEMENT_CLASS_VARIANT(MeasuredGGXBounded, BSDF)
MI_EXPORT_PLUGIN(MeasuredGGXBounded, "Measured bounded GGX material")
NAMESPACE_END(mitsuba)

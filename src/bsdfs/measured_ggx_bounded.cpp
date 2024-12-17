#include <mitsuba/core/properties.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/tensor.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <array>
#include <cmath>
#include "bounded_ggx_vndf.h"

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MeasuredGGXBounded final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES()

    using Warp2D0 = Marginal2D<Float, 0, true>;
    using Warp2D2 = Marginal2D<Float, 2, true>;
    using Warp2D3 = Marginal2D<Float, 3, true>;

    MeasuredGGXBounded(const Properties &props) : Base(props) {
        m_components.push_back(BSDFFlags::GlossyReflection | BSDFFlags::FrontSide);
        m_flags = m_components[0];

        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));

        ref<TensorFile> tf = new TensorFile(file_path);
        using Field = TensorFile::Field;

        const Field &theta_i       = tf->field("theta_i");
        const Field &phi_i         = tf->field("phi_i");
        const Field &alpha         = tf->field("alpha");

        Field spectra, wavelengths;
        const ScalarFloat rgb_wavelengths[3] = { 0, 1, 2 };
        spectra = tf->field("rgb");

        if constexpr (!is_rgb_v<Spectrum>)
            Throw("Measurements in RGB format require the use of a RGB variant of Mitsuba!");

        wavelengths.shape.push_back(3);
        wavelengths.data = rgb_wavelengths;

        if (!(theta_i.shape.size() == 1 &&
              theta_i.dtype == Struct::Type::Float32 &&

              phi_i.shape.size() == 1 &&
              phi_i.dtype == Struct::Type::Float32 &&

              spectra.dtype == Struct::Type::Float32 &&
              spectra.shape.size() == 5 &&
              spectra.shape[0] == phi_i.shape[0] &&
              spectra.shape[1] == theta_i.shape[0] &&
              spectra.shape[2] == 3 &&
              spectra.shape[3] == spectra.shape[4] &&

              alpha.shape.size() == 1 &&
              alpha.shape[0] == 1))
              Throw("Invalid file structure: %s", tf);

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

        m_alpha = std::clamp(static_cast<const float*>(alpha.data)[0], 0.f, 1.f);

        Log(Info, "Loaded material measured bounded GGX material with roughness %f",
            this->m_alpha);
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
        const auto boundedGGX = BoundedGGX<Float, Spectrum>(this->m_alpha);
        const auto& wi = si.wi;

        auto bs = dr::zeros<BSDFSample3f>();
        auto m = boundedGGX.sample(wi, sample2.x(), sample2.y());
        bs.wo = m;
        return { bs, 0.f };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo_, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        UnpolarizedSpectrum spec;
        const auto boundedGGX = BoundedGGX<Float, Spectrum>(this->m_alpha);
        auto u1u2 = boundedGGX.invert(si.wi, wo_);
        spec *= u1u2.x() + u1u2.y();
        return depolarizer<Spectrum>(spec) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo_, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);
        const auto boundedGGX = BoundedGGX<Float, Spectrum>(this->m_alpha);
        return boundedGGX.pdf(si.wi, wo_);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MeasuredBoundedGGX[" << std::endl
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
    Warp2D3 m_spectra;
    bool m_jacobian;
    float m_alpha;
};

MI_IMPLEMENT_CLASS_VARIANT(MeasuredGGXBounded, BSDF)
MI_EXPORT_PLUGIN(MeasuredGGXBounded, "Measured bounded GGX material")
NAMESPACE_END(mitsuba)

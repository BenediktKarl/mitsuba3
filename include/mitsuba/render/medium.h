#pragma once

#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/render/fwd.h>
#include <drjit/vcall.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Medium : public Object {
public:
    MI_IMPORT_TYPES(PhaseFunction, MediumPtr, Sampler, Scene, Texture);

    /// Intersects a ray with the medium's bounding box
    virtual std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const = 0;

    /// Returns the medium's majorant used for delta tracking
    virtual UnpolarizedSpectrum
    get_majorant(const MediumInteraction3f &mi,
                 Mask active = true) const = 0;

    /**
     * Returns the medium's albedo, independently of other quantities.
     * May not be supported by all media.
     *
     * Becomes necessary when we need to evaluate the albedo at a
     * location where sigma_t = 0.
     */
    virtual UnpolarizedSpectrum get_albedo(const MediumInteraction3f &mi,
                                           Mask active = true) const = 0;

    /// Returns the medium's emission at the queried location.
    virtual UnpolarizedSpectrum get_emission(const MediumInteraction3f &mi,
                                             Mask active = true) const = 0;

    /// Returns the medium coefficients Sigma_s, Sigma_n and Sigma_t evaluated
    /// at a given MediumInteraction mi
    virtual std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum,
                       UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active = true) const = 0;

    /**
     * \brief Sample a free-flight distance in the medium.
     *
     * This function samples a (tentative) free-flight distance according to an
     * exponential transmittance. It is then up to the integrator to then decide
     * whether the MediumInteraction corresponds to a real or null scattering
     * event.
     *
     * \param ray      Ray, along which a distance should be sampled
     * \param sample   A uniformly distributed random sample
     * \param channel  The channel according to which we will sample the
     * free-flight distance. This argument is only used when rendering in RGB
     * modes.
     *
     * \return         This method returns a MediumInteraction.
     *                 The MediumInteraction will always be valid,
     *                 except if the ray missed the Medium's bounding box.
     */
    virtual MediumInteraction3f
    sample_interaction(const Ray3f &ray, Float sample,
                       UInt32 channel, Mask active) const;

    // sample a given medium interaction, with the previous and old states
    virtual std::pair<MediumInteraction3f, MediumInteraction3f>
    sample_interaction_twostates(const Ray3f &ray, Float sample,
                       UInt32 channel, Mask active) const;

    virtual UnpolarizedSpectrum
    eval_tr_old(const MediumInteraction3f &mi,
                const SurfaceInteraction3f &si, Mask active) const;

    virtual UnpolarizedSpectrum
    eval_tr_new(const MediumInteraction3f &mi,
                const SurfaceInteraction3f &si, Mask active) const;

    /**
     * Similar to \ref sample_interaction, but ensures that a real interaction
     * is sampled.
     */
    std::pair<MediumInteraction3f, Spectrum>
    sample_interaction_real(const Ray3f &ray, Sampler *sampler, UInt32 channel,
                            Mask active) const;

    /**
     * Sample an interaction with Differential Ratio Tracking.
     * Intended for adjoint integration.
     *
     * Returns the interaction record and a sampling weight.
     */
    std::pair<MediumInteraction3f, Spectrum>
    sample_interaction_drt(const Ray3f &ray, Sampler *sampler, UInt32 channel,
                           Mask active) const;

    /**
     * Sample an interaction with Differential Residual Ratio Tracking.
     * Intended for adjoint integration.
     *
     * Returns the interaction record and a sampling weight.
     */
    std::pair<MediumInteraction3f, Spectrum>
    sample_interaction_drrt(const Ray3f &ray, Sampler *sampler, UInt32 channel,
                            Mask active) const;

    /**
     * \brief Compute the transmittance and PDF
     *
     * This function evaluates the transmittance and PDF of sampling a certain
     * free-flight distance The returned PDF takes into account if a medium
     * interaction occurred (mi.t <= si.t) or the ray left the medium (mi.t >
     * si.t)
     *
     * The evaluated PDF is spectrally varying. This allows to account for the
     * fact that the free-flight distance sampling distribution can depend on
     * the wavelength.
     *
     * \return   This method returns a pair of (Transmittance, PDF).
     *
     */
    virtual std::pair<UnpolarizedSpectrum, UnpolarizedSpectrum>
    eval_tr_and_pdf(const MediumInteraction3f &mi,
                    const SurfaceInteraction3f &si, Mask active) const;

    /**
     * Compute the ray-medium overlap range and prepare a
     * medium interaction to be filled by a sampling routine.
     * Exposed as part of the API to enable testing.
     */
    std::tuple<MediumInteraction3f, Float, Float, Mask>
    prepare_interaction_sampling(const Ray3f &ray, Mask active) const;

    /// Return the phase function of this medium
    const virtual PhaseFunction *phase_function() const;

    /// Return the phase function of this medium
    const virtual PhaseFunction *old_phase_function() const;

    /// Returns whether this specific medium instance uses emitter sampling
    virtual bool use_emitter_sampling() const;

    /// Returns whether this medium is homogeneous
    virtual bool is_homogeneous() const;

    /// Returns whether this medium has a spectrally varying extinction
    virtual bool has_spectral_extinction() const;

    void traverse(TraversalCallback *callback) override;

    /// Return a string identifier
    std::string id() const override { return m_id; }

    /// Set a string identifier
    void set_id(const std::string& id) override { m_id = id; };

    /// Return a human-readable representation of the Medium
    std::string to_string() const override = 0;

    DRJIT_VCALL_REGISTER(Float, mitsuba::Medium)

    MI_DECLARE_CLASS()
protected:
    Medium(const Properties &props);
    virtual ~Medium();

    static Float extract_channel(Spectrum value, UInt32 channel);

protected:
    ref<PhaseFunction> m_phase_function;
    bool m_sample_emitters, m_is_homogeneous, m_has_spectral_extinction;

    /// Identifier (if available)
    std::string m_id;
};

MI_EXTERN_CLASS(Medium)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
//! @{ \name Dr.Jit support for packets of Medium pointers
// -----------------------------------------------------------------------

DRJIT_VCALL_TEMPLATE_BEGIN(mitsuba::Medium)
    DRJIT_VCALL_METHOD(use_emitter_sampling)
    DRJIT_VCALL_METHOD(is_homogeneous)
    DRJIT_VCALL_METHOD(has_spectral_extinction)
    DRJIT_VCALL_METHOD(get_majorant)
    DRJIT_VCALL_METHOD(get_albedo)
    DRJIT_VCALL_METHOD(get_emission)
    DRJIT_VCALL_METHOD(get_scattering_coefficients)
    DRJIT_VCALL_METHOD(intersect_aabb)
    DRJIT_VCALL_METHOD(sample_interaction)
    DRJIT_VCALL_METHOD(phase_function)
    DRJIT_VCALL_METHOD(old_phase_function)
    DRJIT_VCALL_METHOD(eval_tr_old)
    DRJIT_VCALL_METHOD(eval_tr_new)
    DRJIT_VCALL_METHOD(sample_interaction_twostates)
    DRJIT_VCALL_METHOD(sample_interaction_real)
    DRJIT_VCALL_METHOD(sample_interaction_drt)
    DRJIT_VCALL_METHOD(sample_interaction_drrt)
    DRJIT_VCALL_METHOD(eval_tr_and_pdf)
    DRJIT_VCALL_METHOD(prepare_interaction_sampling)
DRJIT_VCALL_TEMPLATE_END(mitsuba::Medium)

//! @}
// -----------------------------------------------------------------------

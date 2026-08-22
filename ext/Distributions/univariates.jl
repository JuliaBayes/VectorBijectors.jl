# General definitions that apply to all univariate distributions.
function Plaice.from_unconstrained_vec(d::D.UnivariateDistribution)
    return Plaice.OnlyWrap(Plaice.inverse(Plaice.scalar_to_scalar_bijector(d)))
end
function Plaice.to_unconstrained_vec(d::D.UnivariateDistribution)
    return Plaice.VectWrap(Plaice.scalar_to_scalar_bijector(d))
end
Plaice.from_vec(::D.UnivariateDistribution) = Plaice.OnlyWrap(Plaice.TypedIdentity())
Plaice.to_vec(::D.UnivariateDistribution) = Plaice.VectWrap(Plaice.TypedIdentity())

# vect_length and unconstrained_vec_length are trivial
Plaice.vec_length(::D.UnivariateDistribution) = 1
Plaice.unconstrained_vec_length(::D.UnivariateDistribution) = 1

# Optics are trivially obtainable.
Plaice.optic_vec(::D.UnivariateDistribution) = [VarNames.Iden()]
Plaice.unconstrained_optic_vec(::D.UnivariateDistribution) = [VarNames.Iden()]

# These continuous distributions have support over the entire real line.
const IDENTITY_UNIVARIATES = Union{
    D.Cauchy,
    D.Chernoff,
    D.Gumbel,
    D.JohnsonSU,
    D.Laplace,
    D.Logistic,
    D.NoncentralT,
    D.Normal,
    D.NormalCanon,
    D.NormalInverseGaussian,
    D.PGeneralizedGaussian,
    D.SkewedExponentialPower,
    D.SkewNormal,
    D.TDist,
    # For discrete distributions, we can't really do any 'transformation'
    D.DiscreteUnivariateDistribution,
}

Plaice.scalar_to_scalar_bijector(::IDENTITY_UNIVARIATES) = TypedIdentity()

# Furthermore, scaling and shifting doesn't affect the support of these distributions
function Plaice.scalar_to_scalar_bijector(
    ::D.AffineDistribution{<:Any,<:Any,<:IDENTITY_UNIVARIATES},
)
    return TypedIdentity()
end

# Bijectors for continuous univariate distributions which have support over the positive (or
# non-negative) real numbers.
const POSITIVE_UNIVARIATES = Union{
    D.BetaPrime,
    D.Chi,
    D.Chisq,
    D.Erlang,
    D.Exponential,
    D.FDist,
    # Wikipedia's definition of the Frechet distribution allows for a location parameter,
    # which could cause its minimum to be nonzero. However, Distributions.jl's `Frechet`
    # does not implement this, so we can lump it in here.
    D.Frechet,
    D.Gamma,
    D.InverseGamma,
    D.InverseGaussian,
    D.Kolmogorov,
    D.Lindley,
    D.LogNormal,
    D.NoncentralChisq,
    D.NoncentralF,
    D.Rayleigh,
    D.Rician,
    D.StudentizedRange,
    D.Weibull,
}
Plaice.scalar_to_scalar_bijector(d::POSITIVE_UNIVARIATES) = Log(minimum(d), 1)

function Plaice.scalar_to_scalar_bijector(
    d::D.AffineDistribution{<:Any,<:Any,<:POSITIVE_UNIVARIATES},
)
    s = sign(D.scale(d))
    return Log(s > 0 ? minimum(d) : maximum(d), s)
end

# This is the fallback option for all other univariate continuous distributions.
function Plaice.scalar_to_scalar_bijector(d::D.ContinuousUnivariateDistribution)
    return Untruncate(minimum(d), maximum(d))
end

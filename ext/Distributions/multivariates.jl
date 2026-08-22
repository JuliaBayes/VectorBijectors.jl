# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

# The AbstractMvNormal abstract type takes care of MvNormal and MvNormalCanon.
Plaice.from_unconstrained_vec(::D.AbstractMvNormal) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::D.AbstractMvNormal) = Plaice.TypedIdentity()
Plaice.unconstrained_vec_length(d::D.AbstractMvNormal) = length(d)
Plaice.unconstrained_optic_vec(d::D.AbstractMvNormal) = Plaice.optic_vec(d)

# NOTE: AbstractMvTDist is not formally exported from Distributions, but this is the only
# 'correct' place to put it
Plaice.from_unconstrained_vec(::D.AbstractMvTDist) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::D.AbstractMvTDist) = Plaice.TypedIdentity()
Plaice.unconstrained_vec_length(d::D.AbstractMvTDist) = length(d)
Plaice.unconstrained_optic_vec(d::D.AbstractMvTDist) = Plaice.optic_vec(d)

# For all multivariate distributions, from_vec and to_vec are just the identity function.
Plaice.from_vec(::D.MultivariateDistribution) = Plaice.TypedIdentity()
Plaice.to_vec(::D.MultivariateDistribution) = Plaice.TypedIdentity()
# which makes vec_length and optic_vec trivial
Plaice.vec_length(d::D.MultivariateDistribution) = length(d)
# TODO(penelopeysm): We assume here that the axes of the distribution are 1:length(d). This
# is not always true, but we don't (yet) have a good way to determine that... If you're
# reading this, check for updates in:
# https://github.com/JuliaStats/Distributions.jl/issues/734
# https://github.com/JuliaStats/Distributions.jl/pull/2009
function Plaice.optic_vec(d::D.MultivariateDistribution)
    return [VarNames.@opticof(_[i]) for i in 1:length(d)]
end

# For discrete multivariate distributions, we really can't transform the 'support'.
Plaice.from_unconstrained_vec(::D.DiscreteMultivariateDistribution) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::D.DiscreteMultivariateDistribution) = Plaice.TypedIdentity()
Plaice.unconstrained_vec_length(d::D.DiscreteMultivariateDistribution) = Plaice.vec_length(d)
Plaice.unconstrained_optic_vec(d::D.DiscreteMultivariateDistribution) = Plaice.optic_vec(d)

# MvLogNormal
Plaice.from_unconstrained_vec(::D.AbstractMvLogNormal) = Plaice.MapExp()
Plaice.to_unconstrained_vec(::D.AbstractMvLogNormal) = Plaice.MapLog()
Plaice.unconstrained_vec_length(d::D.AbstractMvLogNormal) = length(d)
Plaice.unconstrained_optic_vec(d::D.AbstractMvLogNormal) = Plaice.optic_vec(d)

# Simplex distributions
const SIMPLEX_MULTIVARIATES = Union{D.Dirichlet,D.MvLogitNormal}
Plaice.from_unconstrained_vec(::SIMPLEX_MULTIVARIATES) = Plaice.inverse(Plaice.SimplexBijector())
Plaice.to_unconstrained_vec(::SIMPLEX_MULTIVARIATES) = Plaice.SimplexBijector()
Plaice.unconstrained_vec_length(d::SIMPLEX_MULTIVARIATES) = length(d) - 1
function Plaice.unconstrained_optic_vec(d::SIMPLEX_MULTIVARIATES)
    return fill(nothing, Plaice.unconstrained_vec_length(d))
end

# Multivariate distributions which are already unconstrained and independent in
# all dimensions.

Plaice.from_unconstrained_vec(::PM.MvNormal) = Plaice.TypedIdentity()
Plaice.to_unconstrained_vec(::PM.MvNormal) = Plaice.TypedIdentity()
Plaice.unconstrained_vec_length(d::PM.MvNormal) = length(d.μ)
Plaice.unconstrained_optic_vec(d::PM.MvNormal) = Plaice.optic_vec(d)

# For all multivariate distributions, from_vec and to_vec are just the identity function.
Plaice.from_vec(::PM.MultivariateMeasure) = Plaice.TypedIdentity()
Plaice.to_vec(::PM.MultivariateMeasure) = Plaice.TypedIdentity()
# which makes vec_length and optic_vec trivial for MvNormal.
Plaice.vec_length(d::PM.MvNormal) = length(d.μ)
function Plaice.optic_vec(d::PM.MvNormal)
    return [VarNames.@opticof(_[i]) for i in eachindex(d.μ)]
end

# For discrete multivariate distributions, we really can't transform the 'support'.
#
# todo discrete multivariate Not defined in PM yet
#
# Plaice.from_unconstrained_vec(::PM.DiscreteMultivariateMeasure) = Plaice.TypedIdentity()
# Plaice.to_unconstrained_vec(::PM.DiscreteMultivariateMeasure) = Plaice.TypedIdentity()
# Plaice.unconstrained_vec_length(d::PM.DiscreteMultivariateMeasure) = Plaice.vec_length(d)
# Plaice.unconstrained_optic_vec(d::PM.DiscreteMultivariateMeasure) = Plaice.optic_vec(d)

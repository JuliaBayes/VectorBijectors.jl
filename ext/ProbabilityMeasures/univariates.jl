using Plaice: TypedIdentity, Log

Plaice.from_unconstrained_vec(d::PM.UnivariateMeasure) =
    Plaice.OnlyWrap(Plaice.inverse(Plaice.scalar_to_scalar_bijector(d)))
Plaice.to_unconstrained_vec(d::PM.UnivariateMeasure) =
    Plaice.VectWrap(Plaice.scalar_to_scalar_bijector(d))
Plaice.from_vec(::PM.UnivariateMeasure) = Plaice.OnlyWrap(Plaice.TypedIdentity())
Plaice.to_vec(::PM.UnivariateMeasure) = Plaice.VectWrap(Plaice.TypedIdentity())

Plaice.vec_length(::PM.UnivariateMeasure) = 1
Plaice.unconstrained_vec_length(::PM.UnivariateMeasure) = 1

Plaice.optic_vec(::PM.UnivariateMeasure) = [VarNames.Iden()]
Plaice.unconstrained_optic_vec(::PM.UnivariateMeasure) = [VarNames.Iden()]

# Distributions with support over the entire real line.
Plaice.scalar_to_scalar_bijector(::PM.Normal) = Plaice.TypedIdentity()
Plaice.scalar_to_scalar_bijector(::PM.Laplace) = Plaice.TypedIdentity()

# Discrete distributions cannot usefully be transformed to a continuous support.
Plaice.scalar_to_scalar_bijector(::PM.DiscreteUnivariateMeasure) = Plaice.TypedIdentity()

# Distributions with support over the non-negative reals.
Plaice.scalar_to_scalar_bijector(::PM.Exponential) = Plaice.Log(0.0, 1)

# Everything else
function Plaice.scalar_to_scalar_bijector(d::PM.ContinuousUnivariateMeasure)
    return Plaice.Untruncate(minimum(PM.support(d)), maximum(PM.support(d)))
end

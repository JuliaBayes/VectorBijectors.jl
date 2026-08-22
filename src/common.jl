"""
    AbstractBijector

An abstract type for bijective transforms.

Subtypes must implement two methods:

- [`with_logabsdet_jacobian`](@extref ChangesOfVariables.with_logabsdet_jacobian), which
  should return a tuple of (transformed value, log absolute determinant Jacobian) when
  called with a value to transform.
- [`inverse`](@extref InverseFunctions.inverse), which should return another
  `AbstractBijector` that represents the inverse transformation.

The methods

- `(b::AbstractBijector)(x)`, to return just the transformed value, and
- [`logabsdet_jacobian(b::AbstractBijector, x)`](@ref), to return just the log-Jacobian,

are then automatically derived for subtypes of `AbstractBijector` by simply returning either
component of `with_logabsdet_jacobian`'s result. However, custom implementations of those
methods may be provided for efficiency reasons if desired.
"""
abstract type AbstractBijector end

"""
    (b::AbstractBijector)(x)

Return the transformed value of `x` under the bijector `b`.

By default, this returns `first(with_logabsdet_jacobian(b, x))`, but subtypes of
`AbstractBijector` may provide specialised implementations for efficiency.
"""
(b::AbstractBijector)(x) = first(with_logabsdet_jacobian(b, x))

"""
    logabsdet_jacobian(b::AbstractBijector, x)

Return the log absolute determinant of the Jacobian of the bijector `b` at `x`.

By default, this returns `last(with_logabsdet_jacobian(b, x))`, but subtypes of
`AbstractBijector` may provide specialised implementations for efficiency.
"""
logabsdet_jacobian(b::AbstractBijector, x) = last(with_logabsdet_jacobian(b, x))

"""
    ScalarToScalarBijector <: AbstractBijector

An abstract type for bijectors that map scalars to scalars.

Any subtype of this must implement `is_monotonically_increasing` and
`is_monotonically_decreasing`. One of them should be true and one should be false.
"""
abstract type ScalarToScalarBijector <: AbstractBijector end

"""
    TypedIdentity <: ScalarToScalarBijector

The same as `identity`.

This is the appropriate scalar-to-scalar bijector for univariate distributions which have
support over the entire real line. However, note that this can also be used in other
contexts apart from univariate distributions. For example, `TypedIdentity()` is the correct
implementation of `to_vec` and `from_vec` for `MvNormal` distributions.

The problem with using `identity` as a bijector is that ChangesOfVariables.jl defines
`with_logabsdet_jacobian(identity, x) = (x, zero(eltype(x)))`, which can fail if `eltype(x)`
is not a number type! Implementing this allows us to shortcircuit that definition and return
a sensible result (i.e. a Bool) even if `x` is not a numeric vector. Note that we
intentionally do not return `0.0::Float64` in such cases as that can cause unwanted
promotion.
"""
struct TypedIdentity <: ScalarToScalarBijector end
is_monotonically_increasing(::TypedIdentity) = true
is_monotonically_decreasing(::TypedIdentity) = false
(::TypedIdentity)(x) = x
function with_logabsdet_jacobian(::TypedIdentity, x::AbstractArray{T}) where {T<:Number}
    return (x, zero(T))
end
with_logabsdet_jacobian(::TypedIdentity, x::T) where {T<:Number} = (x, zero(T))
with_logabsdet_jacobian(::TypedIdentity, x) = (x, false)
inverse(x::TypedIdentity) = x

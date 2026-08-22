"""
    OnlyWrap{B<:ScalarToScalarBijector}

Wrap a bijector `B` which transforms scalars to scalars, into a bijector that transforms
vectors of length one to scalars.
"""
struct OnlyWrap{B<:ScalarToScalarBijector} <: AbstractBijector
    bijector::B
end
# Use sum(x) instead of x[] to avoid scalar indexing.
(w::OnlyWrap)(x) = w.bijector(sum(x))
function with_logabsdet_jacobian(w::OnlyWrap, x::AbstractVector)
    return with_logabsdet_jacobian(w.bijector, sum(x))
end
inverse(w::OnlyWrap) = VectWrap(inverse(w.bijector))

"""
    VectWrap{B<:ScalarToScalarBijector}

Wrap a bijector `B` which transforms scalars to scalars, into a bijector that transforms
scalars to vectors of length one.
"""
struct VectWrap{B<:ScalarToScalarBijector} <: AbstractBijector
    bijector::B
end
(w::VectWrap)(x) = [w.bijector(x)]
function with_logabsdet_jacobian(w::VectWrap, x::Number)
    y, ladj = with_logabsdet_jacobian(w.bijector, x)
    return ([y], ladj)
end
inverse(w::VectWrap) = OnlyWrap(inverse(w.bijector))

"""
    Plaice.scalar_to_scalar_bijector(dist)

The Plaice interface is intended to map samples to vectors. However, for univariate
distributions, the 'vectorisation' part of this is trivial (we only need to convert a scalar
to a vector of length one, and vice versa). Therefore, this function is provided to allow
users to specify the 'interesting' part of the transformation, which is the function that
maps values to unconstrained space.

Overloading this function for a univariate distribution is sufficient to implement the
entire Plaice interface for that distribution.

There are three scalar-to-scalar bijectors that are exported, which should be enough for any
univariate distribution:

- [`Plaice.TypedIdentity`](@ref)
- [`Plaice.Log`](@ref)
- [`Plaice.Untruncate`](@ref)

If you need a different scalar-to-scalar bijector, please open an issue.
"""
function scalar_to_scalar_bijector end

# Generic definitions for matrix distributions.

# Somehow, ChangesOfVariables doesn't have a logjac implemented for `vec`, so we need to
# wrap it.
struct Vec{N} <: AbstractBijector
    size::NTuple{N,Int}
end
(::Vec)(x::AbstractArray) = vec(x)
inverse(v::Vec) = Reshape(v.size)

function with_logabsdet_jacobian(::Vec, x::AbstractArray{T,N}) where {T<:Number,N}
    return vec(x), zero(T)
end
function with_logabsdet_jacobian(::Vec, x::AbstractArray)
    return vec(x), false
end

struct Reshape{N} <: AbstractBijector
    size::NTuple{N,Int}
end
(r::Reshape)(x::AbstractArray) = reshape(x, r.size)
inverse(r::Reshape) = Vec(r.size)
function with_logabsdet_jacobian(r::Reshape, x::AbstractArray{T,N}) where {T<:Number,N}
    return reshape(x, r.size), zero(T)
end
function with_logabsdet_jacobian(r::Reshape, x::AbstractArray)
    return reshape(x, r.size), false
end

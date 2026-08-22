# This file handles things like `reshape(dist, newshape...)` or `vec(dist)` by wrapping
# the original bijector and composing it with a reshape operation. It can be inefficient in
# some edge cases (for example, `vec(Normal())` will unwrap and rewrap the value in an
# array), but these cases seem meaningless enough that we can ignore them.

# When reshaping univariate distributions, `original_size` get stored as `()`. If we naively
# use `reshape` we will get a 0-dimensional array out, which is not the same as the scalar
# that `rand(dist)` returns. So we need to use this helper function.
_reshape_or_only(x::AbstractArray, ::Tuple{}) = sum(x)
_reshape_or_only(x, ::Tuple{}) = x
_reshape_or_only(x::AbstractArray, sz) = reshape(x, sz)
# This method handles the case where we need to 'reshape' a scalar into an array
_reshape_or_only(x, sz) = reshape([x], sz)

"""
    ReshapeWrapper(reshaped_size::Tuple, original_size::Tuple, bijector::Bijector)

Here, `original_size` is equal to `size(dist)`, and `reshaped_size` is the size after
reshaping. The wrapped `bijector` converts a sample from `original_size` to a vector or
unconstrained vector. Thus, ReshapeWrapper must:

- first convert the input from `reshaped_size` to `original_size` via `reshape`
- then apply the wrapped `bijector`.
"""
struct ReshapeWrapper{N1,N2,T1<:NTuple{N1,Int},T2<:NTuple{N2,Int},B} <: AbstractBijector
    reshaped_size::T1
    original_size::T2
    bijector::B
end
function Base.:(==)(r1::ReshapeWrapper, r2::ReshapeWrapper)
    return (r1.reshaped_size == r2.reshaped_size) & (r1.original_size == r2.original_size) &
           (r1.bijector == r2.bijector)
end
function Base.isequal(r1::ReshapeWrapper, r2::ReshapeWrapper)
    return isequal(r1.reshaped_size, r2.reshaped_size) &&
           isequal(r1.original_size, r2.original_size) &&
           isequal(r1.bijector, r2.bijector)
end
function with_logabsdet_jacobian(
    r::ReshapeWrapper{N1},
    rx::AbstractArray{T,N1},
) where {T,N1}
    x = _reshape_or_only(rx, r.original_size)
    return with_logabsdet_jacobian(r.bijector, x)
end
function inverse(r::ReshapeWrapper)
    return InvReshapeWrapper(r.reshaped_size, r.original_size, inverse(r.bijector))
end

"""
    InvReshapeWrapper(reshaped_size::Tuple, original_size::Tuple, inv_bijector::Bijector)

This is the inverse of ReshapeWrapper. It does a similar thing to `ReshapeWrapper`, but in a
different order, since it must apply `inv_bijector` first before reshaping the output.
"""
struct InvReshapeWrapper{N1,N2,T1<:NTuple{N1,Int},T2<:NTuple{N2,Int},B} <: AbstractBijector
    reshaped_size::T1
    original_size::T2
    inv_bijector::B
end
function Base.:(==)(r1::InvReshapeWrapper, r2::InvReshapeWrapper)
    return (r1.reshaped_size == r2.reshaped_size) & (r1.original_size == r2.original_size) &
           (r1.inv_bijector == r2.inv_bijector)
end
function Base.isequal(r1::InvReshapeWrapper, r2::InvReshapeWrapper)
    return isequal(r1.reshaped_size, r2.reshaped_size) &&
           isequal(r1.original_size, r2.original_size) &&
           isequal(r1.inv_bijector, r2.inv_bijector)
end
function with_logabsdet_jacobian(r::InvReshapeWrapper, x::AbstractVector)
    rx, ladj = with_logabsdet_jacobian(r.inv_bijector, x)
    rx_reshaped = _reshape_or_only(rx, r.reshaped_size)
    return (rx_reshaped, ladj)
end
function inverse(r::InvReshapeWrapper)
    return ReshapeWrapper(r.reshaped_size, r.original_size, inverse(r.inv_bijector))
end

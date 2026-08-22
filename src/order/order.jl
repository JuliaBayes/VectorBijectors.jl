# For JointOrderStatistics, we need to map the original bijector over each element. That
# gives us an ordered vector, but we are not done there: we need to then map that ordered
# vector back to a regular (unordered) vector. This uses something similar to
# OrderedBijector(), but we reimplement it here to avoid extra allocations.
struct JointOrderWrap{B<:ScalarToScalarBijector} <: AbstractBijector
    bijector::B
end
function with_logabsdet_jacobian(m::JointOrderWrap, x::AbstractVector{T}) where {T<:Number}
    # `x` is always an ordered vector. Sometimes, mapping m.bijector over x doesn't give
    # an ordered vector: it could give a *reverse* ordered vector, if m.bijector performs
    # a sign flip (i.e., is monotonically decreasing). In that case, we need to undo the
    # sign flip to get back to the original ordering.
    s = is_monotonically_decreasing(m.bijector) ? -1 : 1
    logjac = zero(T)
    y = similar(x)
    for i in eachindex(x)
        yi, lj = with_logabsdet_jacobian(m.bijector, x[i])
        y[i] = s * yi
        logjac += lj
    end
    # Now `y` will definitely be ordered. To transform this to an unordered vector, we
    # see that y[1] is already fine (it ranges from -Inf to Inf), but y[2] has a lower
    # bound of y[1]. So we need to shift it down by y[1] and take the logarithm.
    # Similarly, y[3] has a lower bound of y[2], etc. etc.
    if length(x) > 1
        shift = y[1]
        for i in eachindex(y)[2:end]
            temp = y[i]
            y[i] = log(temp - shift)
            logjac -= y[i]
            shift = temp
        end
    end
    return y, logjac
end
inverse(m::JointOrderWrap) = InverseJointOrderWrap(inverse(m.bijector))

struct InverseJointOrderWrap{B<:ScalarToScalarBijector} <: AbstractBijector
    bijector::B
end
function with_logabsdet_jacobian(
    m::InverseJointOrderWrap,
    y::AbstractVector{T},
) where {T<:Number}
    # First, we need to undo the logarithmic transformations to get back to the ordered
    # vector.
    logjac = zero(T)
    x = copy(y)
    if length(y) > 1
        for i in eachindex(y)[2:end]
            temp = x[i]
            x[i] = exp(temp) + x[i-1]
            logjac += temp
        end
    end
    s = is_monotonically_decreasing(m.bijector) ? -1 : 1
    # Now `x` is an ordered vector. We need to apply the signflip if necessary, and then
    # map the inner bijector.
    for i in eachindex(x)
        xi, lj = with_logabsdet_jacobian(m.bijector, s * x[i])
        x[i] = xi
        logjac += lj
    end
    return x, logjac
end
inverse(m::InverseJointOrderWrap) = JointOrderWrap(inverse(m.bijector))

struct MapLog <: AbstractBijector end
(::MapLog)(x) = map(log, x)
function with_logabsdet_jacobian(::MapLog, x::AbstractArray{T}) where {T<:Number}
    y = map(log, x)
    return (y, -sum(y))
end
inverse(::MapLog) = MapExp()

struct MapExp <: AbstractBijector end
(::MapExp)(x) = map(exp, x)
function with_logabsdet_jacobian(::MapExp, x::AbstractArray{T}) where {T<:Number}
    y = map(exp, x)
    return (y, sum(x))
end
inverse(::MapExp) = MapLog()

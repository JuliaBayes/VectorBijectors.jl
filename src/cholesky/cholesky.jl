using LinearAlgebra: LinearAlgebra as LA

function _get_cartesian_indices(n::Int, uplo::Char)
    if uplo == 'U'
        return [(i, j) for j in 1:n for i in 1:j]
    else
        return [(i, j) for j in 1:n for i in j:n]
    end
end

struct CholeskyVec <: AbstractBijector
    n::Int
    uplo::Char
end
function (c::CholeskyVec)(x::LA.Cholesky)
    cartesian_indices = _get_cartesian_indices(c.n, c.uplo)
    return [x.UL[i, j] for (i, j) in cartesian_indices]
end
function with_logabsdet_jacobian(c::CholeskyVec, x::LA.Cholesky{T}) where {T<:Number}
    return (c(x), zero(T))
end

struct CholeskyUnVec <: AbstractBijector
    n::Int
    uplo::Char
end
inverse(c::CholeskyVec) = CholeskyUnVec(c.n, c.uplo)
inverse(c::CholeskyUnVec) = CholeskyVec(c.n, c.uplo)

function (c::CholeskyUnVec)(xvec::AbstractVector{T}) where {T<:Number}
    x = if c.uplo == 'U'
        LA.Cholesky(LA.UpperTriangular(zeros(T, c.n, c.n)))
    else
        LA.Cholesky(LA.LowerTriangular(zeros(T, c.n, c.n)))
    end
    cartesian_indices = _get_cartesian_indices(c.n, c.uplo)
    for (idx, (i, j)) in enumerate(cartesian_indices)
        x.UL[i, j] = xvec[idx]
    end
    return x
end
function with_logabsdet_jacobian(c::CholeskyUnVec, x::AbstractVector{T}) where {T<:Number}
    return (c(x), zero(T))
end

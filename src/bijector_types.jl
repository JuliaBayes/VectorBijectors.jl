using LinearAlgebra: LinearAlgebra
using LogExpFunctions: LogExpFunctions

# ---- Helpers ----

_eps(::Type{T}) where {T<:Number} = T(eps(T))
_eps(::Type{Real}) = eps(Float64)
_eps(::Type{T}) where {T<:Integer} = eps(Float64)

function _clamp(x, a, b)
    T = promote_type(typeof(x), typeof(a), typeof(b))
    clamped_x = ifelse(x < a, convert(T, a), ifelse(x > b, convert(T, b), x))
    return clamped_x
end

# ---- is_monotonically_increasing / is_monotonically_decreasing ----

"""
    is_monotonically_increasing(f)

Returns `true` if `f` is monotonically increasing.
"""
is_monotonically_increasing(f) = false
is_monotonically_increasing(::typeof(identity)) = true

"""
    is_monotonically_decreasing(f)

Returns `true` if `f` is monotonically decreasing.
"""
is_monotonically_decreasing(f) = false
is_monotonically_decreasing(::typeof(identity)) = false

# ---- Utility functions from Bijectors.jl/src/utils.jl ----

# # Because `ReverseDiff` does not play well with structural matrices.
lower_triangular(A::AbstractMatrix) = convert(typeof(A), LinearAlgebra.LowerTriangular(A))
upper_triangular(A::AbstractMatrix) = convert(typeof(A), LinearAlgebra.UpperTriangular(A))

function pd_from_upper(X)
    U = upper_triangular(X)
    return U' * U
end

# HACK: Allows us to define custom chain rules while we wait for upstream fixes.
transpose_eager(X::AbstractMatrix) = permutedims(X)

cholesky_upper(X::AbstractMatrix) =
    upper_triangular(parent(LinearAlgebra.cholesky(LinearAlgebra.Hermitian(X)).U))
cholesky_upper(X::LinearAlgebra.Cholesky) = X.U

cholesky_lower(X::AbstractMatrix) =
    lower_triangular(parent(LinearAlgebra.cholesky(LinearAlgebra.Hermitian(X, :L)).L))
cholesky_lower(X::LinearAlgebra.Cholesky) = X.L

#      n * (n - 1) / 2 = d
# ⟺       n^2 - n - 2d = 0
# ⟹                  n = (1 + sqrt(1 + 8d)) / 2
_triu1_dim_from_length(d) = (1 + isqrt(1 + 8d)) ÷ 2

function vec_to_triu1_row_index(idx)
    # Assumes that vector was saved in a column-major order
    # and that vector is one-based indexed.
    M = _triu1_dim_from_length(idx - 1)
    return idx - (M * (M - 1) ÷ 2)
end

# ---- SimplexBijector ----

####################
# Simplex bijector #
####################
struct SimplexBijector <: AbstractBijector end

(b::SimplexBijector)(x) = _simplex_bijector(x, b)
inverse(::SimplexBijector) = InverseSimplexBijector()

function with_logabsdet_jacobian(b::SimplexBijector, x)
    return _simplex_bijector(x, b), _logabsdetjac_simplex(b, x)
end

struct InverseSimplexBijector <: AbstractBijector end
inverse(::InverseSimplexBijector) = SimplexBijector()

(::InverseSimplexBijector)(y) = _simplex_inv_bijector(y)

function with_logabsdet_jacobian(::InverseSimplexBijector, y)
    x = _simplex_inv_bijector(y)
    return x, -_logabsdetjac_simplex(SimplexBijector(), x)
end

function _simplex_bijector(x::AbstractArray, b::SimplexBijector)
    sz = size(x)
    K = size(x, 1)
    y = similar(x, Base.setindex(sz, K - 1, 1))
    _simplex_bijector!(y, x, b)
    return y
end

# Vector implementation.
function _simplex_bijector!(y, x::AbstractVector, ::SimplexBijector)
    K = length(x)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(x)
    ϵ = _eps(T)
    sum_tmp = zero(T)
    z = x[1] * (one(T) - 2ϵ) + ϵ # z ∈ [ϵ, 1-ϵ]
    y[1] = LogExpFunctions.logit(z) + log(T(K - 1))
    for k in 2:(K-1)
        sum_tmp += x[k-1]
        # z ∈ [ϵ, 1-ϵ]
        # x[k] = 0 && sum_tmp = 1 -> z ≈ 1
        z = (x[k] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
        y[k] = LogExpFunctions.logit(z) + log(T(K - k))
    end
    return y
end

# Matrix implementation.
function _simplex_bijector!(Y, X::AbstractMatrix, ::SimplexBijector)
    K, N = size(X, 1), size(X, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(X)
    ϵ = _eps(T)
    for n in 1:size(X, 2)
        sum_tmp = zero(T)
        z = X[1, n] * (one(T) - 2ϵ) + ϵ
        Y[1, n] = LogExpFunctions.logit(z) + log(T(K - 1))
        for k in 2:(K-1)
            sum_tmp += X[k-1, n]
            z = (X[k, n] + ϵ) * (one(T) - 2ϵ) / ((one(T) + ϵ) - sum_tmp)
            Y[k, n] = LogExpFunctions.logit(z) + log(T(K - k))
        end
    end

    return Y
end

function _simplex_inv_bijector(y)
    sz = size(y)
    K = sz[1] + 1
    x = similar(y, Base.setindex(sz, K, 1))
    _simplex_inv_bijector!(x, y)
    return x
end

function _simplex_inv_bijector!(x, y::AbstractVector)
    K = length(y) + 1
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(y)
    ϵ = _eps(T)
    z = LogExpFunctions.logistic(y[1] - log(T(K - 1)))
    x[1] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
    sum_tmp = zero(T)
    for k in 2:(K-1)
        z = LogExpFunctions.logistic(y[k] - log(T(K - k)))
        sum_tmp += x[k-1]
        x[k] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
    end
    sum_tmp += x[K-1]
    x[K] = _clamp(one(T) - sum_tmp, 0, 1)
    return x
end

function _simplex_inv_bijector!(X, Y::AbstractMatrix)
    K, N = size(Y, 1) + 1, size(Y, 2)
    @assert K > 1 "x needs to be of length greater than 1"
    T = eltype(Y)
    ϵ = _eps(T)
    for n in 1:size(X, 2)
        sum_tmp, z = zero(T), LogExpFunctions.logistic(Y[1, n] - log(T(K - 1)))
        X[1, n] = _clamp((z - ϵ) / (one(T) - 2ϵ), 0, 1)
        for k in 2:(K-1)
            z = LogExpFunctions.logistic(Y[k, n] - log(T(K - k)))
            sum_tmp += X[k-1, n]
            X[k, n] = _clamp(((one(T) + ϵ) - sum_tmp) / (one(T) - 2ϵ) * z - ϵ, 0, 1)
        end
        sum_tmp += X[K-1, n]
        X[K, n] = _clamp(one(T) - sum_tmp, 0, 1)
    end

    return X
end

function _logabsdetjac_simplex(b::SimplexBijector, x::AbstractVector{T}) where {T}
    ϵ = _eps(T)
    lp = zero(T)

    K = length(x)

    sum_tmp = zero(eltype(x))
    z = x[1]
    lp += log(max(z, ϵ)) + log(max(one(T) - z, ϵ))
    for k in 2:(K-1)
        sum_tmp += x[k-1]
        z = x[k] / max(one(T) - sum_tmp, ϵ)
        lp += log(max(z, ϵ)) + log(max(one(T) - z, ϵ)) + log(max(one(T) - sum_tmp, ϵ))
    end

    return -lp
end

# Needed to avoid falling back to `with_logabsdet_jacobian` for matrix inputs.
function _logabsdetjac_simplex(b::SimplexBijector, x::AbstractMatrix{<:Number})
    return sum(Base.Fix1(_logabsdetjac_simplex, b), eachcol(x))
end

# ---- Correlation / Cholesky helpers ----

"""
    _link_chol_lkj(w)

Link function for cholesky factor.

An alternative and maybe more efficient implementation was considered:

```
for i=2:K, j=(i+1):K
    z[i, j] = (w[i, j] / w[i-1, j]) * (z[i-1, j] / sqrt(1 - z[i-1, j]^2))
end
```

But this implementation will not work when w[i-1, j] = 0.
Though it is a zero measure set, unit matrix initialization will not work.

For equivalence, following explanations is given by @torfjelde:

For `(i, j)` in the loop below, we define

    z₍ᵢ₋₁, ⱼ₎ = w₍ᵢ₋₁,ⱼ₎ * ∏ₖ₌₁ⁱ⁻² (1 / √(1 - z₍ₖ,ⱼ₎²))

and so

    z₍ᵢ,ⱼ₎ = w₍ᵢ,ⱼ₎ * ∏ₖ₌₁ⁱ⁻¹ (1 / √(1 - z₍ₖ,ⱼ₎²))
            = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (∏ₖ₌₁ⁱ⁻² 1 / √(1 - z₍ₖ,ⱼ₎²))
            = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (w₍ᵢ₋₁,ⱼ₎ * ∏ₖ₌₁ⁱ⁻² 1 / √(1 - z₍ₖ,ⱼ₎²)) / w₍ᵢ₋₁,ⱼ₎
            = (w₍ᵢ,ⱼ₎ * / √(1 - z₍ᵢ₋₁,ⱼ₎²)) * (z₍ᵢ₋₁,ⱼ₎ / w₍ᵢ₋₁,ⱼ₎)
            = (w₍ᵢ,ⱼ₎ / w₍ᵢ₋₁,ⱼ₎) * (z₍ᵢ₋₁,ⱼ₎ / √(1 - z₍ᵢ₋₁,ⱼ₎²))

which is the above implementation.
"""
function _link_chol_lkj(W::AbstractMatrix)
    K = LinearAlgebra.checksquare(W)

    y = similar(W) # W is upper triangular.
    # Some zero filling can be avoided. Though diagnoal is still needed to be filled with zero.

    for j in 1:K
        remainder_sq = W[j, j]^2
        for i in (j-1):-1:1
            z = W[i, j] / sqrt(remainder_sq)
            y[i, j] = asinh(z)
            remainder_sq += W[i, j]^2
        end
        for i in j:K
            y[i, j] = 0
        end
    end

    return y
end

function _link_chol_lkj_from_upper(W::AbstractMatrix)
    K = LinearAlgebra.checksquare(W)
    N = ((K - 1) * K) ÷ 2   # {K \choose 2} free parameters

    y = similar(W, N)

    starting_idx = 1
    for j in 2:K
        y[starting_idx] = atanh(W[1, j])
        starting_idx += 1
        remainder_sq = W[j, j]^2
        for i in (j-1):-1:2
            idx = starting_idx + i - 2
            z = W[i, j] / sqrt(remainder_sq)
            y[idx] = asinh(z)
            remainder_sq += W[i, j]^2
        end
        starting_idx += length((j-1):-1:2)
    end

    return y
end

_link_chol_lkj_from_lower(W::AbstractMatrix) = _link_chol_lkj_from_upper(transpose_eager(W))

"""
    _inv_link_chol_lkj(y)

Inverse link function for cholesky factor.
"""
function _inv_link_chol_lkj(Y::AbstractMatrix)
    LinearAlgebra.require_one_based_indexing(Y)
    K = LinearAlgebra.checksquare(Y)

    W = similar(Y)
    T = float(eltype(W))
    logJ = zero(T)

    for j in 1:K
        log_remainder = zero(T)  # log of proportion of unit vector remaining
        for i in 1:(j-1)
            z = tanh(Y[i, j])
            W[i, j] = z * exp(log_remainder)
            log_remainder -= LogExpFunctions.logcosh(Y[i, j])
            logJ += log_remainder
        end
        logJ += log_remainder
        W[j, j] = exp(log_remainder)
        for i in (j+1):K
            W[i, j] = 0
        end
    end

    return W, logJ
end

function _inv_link_chol_lkj(y::AbstractVector)
    LinearAlgebra.require_one_based_indexing(y)
    K = _triu1_dim_from_length(length(y))

    W = similar(y, K, K)
    T = float(eltype(W))
    logJ = zero(T)

    z_vec = map(tanh, y)
    lc_vec = map(LogExpFunctions.logcosh, y)

    idx = 1
    for j in 1:K
        log_remainder = zero(T)  # log of proportion of unit vector remaining
        for i in 1:(j-1)
            z = z_vec[idx]
            W[i, j] = z * exp(log_remainder)
            log_remainder -= lc_vec[idx]
            logJ += log_remainder
            idx += 1
        end
        logJ += log_remainder
        W[j, j] = exp(log_remainder)
        for i in (j+1):K
            W[i, j] = 0
        end
    end

    return W, logJ
end

function _logabsdetjac_inv_corr(Y::AbstractMatrix)
    K = LinearAlgebra.checksquare(Y)

    result = float(zero(eltype(Y)))
    for j in 2:K, i in 1:(j-1)
        result -= (K - i + 1) * LogExpFunctions.logcosh(Y[i, j])
    end
    return result
end

function _logabsdetjac_inv_corr(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    result = float(zero(eltype(y)))
    for (i, y_i) in enumerate(y)
        row_idx = vec_to_triu1_row_index(i)
        result -= (K - row_idx + 1) * LogExpFunctions.logcosh(y_i)
    end
    return result
end

function _logabsdetjac_inv_chol(y::AbstractVector)
    K = _triu1_dim_from_length(length(y))

    result = float(zero(eltype(y)))
    idx = 1
    for j in 2:K
        tmp = zero(result)
        for _ in 1:(j-1)
            logcoshy = LogExpFunctions.logcosh(y[idx])
            tmp -= logcoshy
            result += tmp - logcoshy
            idx += 1
        end
    end

    return result
end

# ---- VecCorrBijector ----

"""
    VecCorrBijector

A bijector to transform a correlation matrix to an unconstrained vector.

# Reference
https://mc-stan.org/docs/reference-manual/correlation-matrix-transform.html
"""
struct VecCorrBijector <: AbstractBijector end

(b::VecCorrBijector)(X) = _link_chol_lkj_from_upper(cholesky_upper(X))
inverse(::VecCorrBijector) = InverseVecCorrBijector()

function with_logabsdet_jacobian(b::VecCorrBijector, x)
    y = b(x)
    return y, -_logabsdetjac_inv_corr(y)
end

struct InverseVecCorrBijector <: AbstractBijector end
inverse(::InverseVecCorrBijector) = VecCorrBijector()

function with_logabsdet_jacobian(::InverseVecCorrBijector, y)
    U_logJ = _inv_link_chol_lkj(y)
    # workaround for tuples that don't support iteration in certain AD backends
    U, logJ = U_logJ[1], U_logJ[2]
    K = size(U, 1)
    for j in 2:(K-1)
        logJ += (K - j) * log(U[j, j])
    end
    return pd_from_upper(U), logJ
end

# ---- VecCholeskyBijector ----

"""
    VecCholeskyBijector

A bijector to transform a Cholesky factor of a correlation matrix to an unconstrained vector.

# Fields
- mode :`Symbol`. Controls the inverse tranformation :
    - if `mode === :U` returns a `LinearAlgebra.Cholesky` holding the `UpperTriangular` factor
    - if `mode === :L` returns a `LinearAlgebra.Cholesky` holding the `LowerTriangular` factor

# Reference
https://mc-stan.org/docs/reference-manual/cholesky-factors-of-correlation-matrices-1
"""
struct VecCholeskyBijector <: AbstractBijector
    mode::Symbol
    function VecCholeskyBijector(uplo)
        s = Symbol(uplo)
        if (s === :U) || (s === :L)
            new(s)
        else
            throw(
                ArgumentError(
                    "mode must be either :U (upper triangular) or :L (lower triangular)",
                ),
            )
        end
    end
end

function (b::VecCholeskyBijector)(X)
    return if b.mode === :U
        _link_chol_lkj_from_upper(cholesky_upper(X))
    else # No need to check for === :L, as it is checked in the VecCholeskyBijector constructor.
        _link_chol_lkj_from_lower(cholesky_lower(X))
    end
end

inverse(b::VecCholeskyBijector) = InverseVecCholeskyBijector(b.mode)

function with_logabsdet_jacobian(b::VecCholeskyBijector, x)
    y = b(x)
    return y, -_logabsdetjac_inv_chol(y)
end

struct InverseVecCholeskyBijector <: AbstractBijector
    mode::Symbol
end
inverse(b::InverseVecCholeskyBijector) = VecCholeskyBijector(b.mode)

function with_logabsdet_jacobian(b::InverseVecCholeskyBijector, y)
    factors, logJ = _inv_link_chol_lkj(y)
    if b.mode === :U
        # This Cholesky constructor is compatible with Julia v1.6
        # for later versions Cholesky(::UpperTriangular) works
        return LinearAlgebra.Cholesky(factors, 'U', 0), logJ
    else # No need to check for === :L, as it is checked in the VecCholeskyBijector constructor.
        # HACK: Need to make materialize the transposed matrix to avoid numerical instabilities.
        # If we don't, the return-type can be both `Matrix` and `Transposed`.
        return LinearAlgebra.Cholesky(transpose_eager(factors), 'L', 0), logJ
    end
end

function Plaice.optic_vec(d::D.LKJCholesky)
    n = first(size(d))
    sym = if d.uplo == 'U'
        :U
    else
        :L
    end
    return [
        VarNames.@opticof(_.$sym[i, j]) for (i, j) in Plaice._get_cartesian_indices(n, d.uplo)
    ]
end

Plaice.from_vec(d::D.LKJCholesky) = Plaice.CholeskyUnVec(first(size(d)), d.uplo)
Plaice.to_vec(d::D.LKJCholesky) = Plaice.CholeskyVec(first(size(d)), d.uplo)
function Plaice.vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
Plaice.from_unconstrained_vec(d::D.LKJCholesky) = Plaice.inverse(Plaice.VecCholeskyBijector(d.uplo))
Plaice.to_unconstrained_vec(d::D.LKJCholesky) = Plaice.VecCholeskyBijector(d.uplo)
function Plaice.unconstrained_vec_length(d::D.LKJCholesky)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
function Plaice.unconstrained_optic_vec(d::D.LKJCholesky)
    return fill(nothing, Plaice.unconstrained_vec_length(d))
end

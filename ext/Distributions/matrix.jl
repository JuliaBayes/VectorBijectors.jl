Plaice.to_vec(d::D.MatrixDistribution) = Plaice.Vec(size(d))
Plaice.from_vec(d::D.MatrixDistribution) = Plaice.Reshape(size(d))
Plaice.vec_length(d::D.MatrixDistribution) = prod(size(d))
function Plaice.optic_vec(d::D.MatrixDistribution)
    return map(c -> VarNames.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end

# MatrixNormal and MatrixTDist are trivial since all their components are already
# unconstrained.

const UnconsMatrixDist = Union{D.MatrixNormal,D.MatrixTDist}

Plaice.to_unconstrained_vec(d::UnconsMatrixDist) = Plaice.Vec(size(d))
Plaice.from_unconstrained_vec(d::UnconsMatrixDist) = Plaice.Reshape(size(d))
Plaice.unconstrained_vec_length(d::UnconsMatrixDist) = prod(size(d))
function Plaice.unconstrained_optic_vec(d::UnconsMatrixDist)
    return map(c -> VarNames.Index(c.I, (;)), vec(CartesianIndices(size(d))))
end

# TODO(penelopeysm): MatrixBeta also generates positive definite matrices. However, it is
# even more specific than Wishart/InverseWishart in that it generates positive definite
# matrices `M` such that `I - M` is also positive definite. This means that the
# transformation implemented here is not suitable for MatrixBeta, as
# from_unconstrained_vec(d)(randn(...)) may not be in the support of MatrixBeta, and thus sampling
# from a unconstrained vector with e.g. NUTS may fail. Hence, we do not include MatrixBeta here.
const PDMatrixDistribution = Union{D.Wishart,D.InverseWishart}

Plaice.from_unconstrained_vec(d::PDMatrixDistribution) = Plaice.InvPosDef(first(size(d)))
Plaice.to_unconstrained_vec(d::PDMatrixDistribution) = Plaice.PosDef(first(size(d)))
function Plaice.unconstrained_vec_length(d::PDMatrixDistribution)
    n = first(size(d))
    return div(n * (n + 1), 2)
end
Plaice.unconstrained_optic_vec(d::PDMatrixDistribution) =
    fill(nothing, Plaice.unconstrained_vec_length(d))

# LKJ correlation matrices.
#
# TODO(penelopeysm) VecCorrBijector has a few bugs. Look at the issue tracker.

Plaice.from_unconstrained_vec(::D.LKJ) = Plaice.inverse(Plaice.VecCorrBijector())
Plaice.to_unconstrained_vec(::D.LKJ) = Plaice.VecCorrBijector()
function Plaice.unconstrained_vec_length(d::D.LKJ)
    n = first(size(d))
    return div(n * (n - 1), 2)
end
Plaice.unconstrained_optic_vec(d::D.LKJ) = fill(nothing, Plaice.unconstrained_vec_length(d))

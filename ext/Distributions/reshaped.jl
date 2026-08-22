# Need some special cases for optics.
const ReshapedUnivariateDistribution =
    D.ReshapedDistribution{<:Any,<:D.ValueSupport,<:D.UnivariateDistribution}

Plaice.to_vec(d::D.ReshapedDistribution) =
    Plaice.ReshapeWrapper(size(d), size(d.dist), Plaice.to_vec(d.dist))
function Plaice.from_vec(d::D.ReshapedDistribution)
    return Plaice.InvReshapeWrapper(size(d), size(d.dist), Plaice.from_vec(d.dist))
end
Plaice.vec_length(d::D.ReshapedDistribution) = Plaice.vec_length(d.dist)

function Plaice.to_unconstrained_vec(d::D.ReshapedDistribution)
    return Plaice.ReshapeWrapper(size(d), size(d.dist), Plaice.to_unconstrained_vec(d.dist))
end
function Plaice.from_unconstrained_vec(d::D.ReshapedDistribution)
    return Plaice.InvReshapeWrapper(size(d), size(d.dist), Plaice.from_unconstrained_vec(d.dist))
end
Plaice.unconstrained_vec_length(d::D.ReshapedDistribution) = Plaice.unconstrained_vec_length(d.dist)

# optic_vec requires some care. We can't just reuse the original distribution's optics,
# i.e., `optic_vec(d) = optic_vec(d.dist)` because the axes may have changed due to
# reshaping. In some cases it might just happen to work (e.g. if `d.dist` is a multivariate
# distribution, `optic_vec(d.dist)` would return [_[1], _[2], ...] which would work on any
# AbstractArray because of linear indexing. However, that isn't general.
#
# Broadly speaking, we need to map the original distribution's optics through the reshape
# operation. That is, if `optic_vec(d.dist)` returns `[_[i1...], _[i2...], ...]` where `i1`
# and `i2` are tuples of indices into the original distribution's array, we need to return
# `[_[j1...], _[j2...], ...]` where `j1` is the indices that `i1` would be mapped to by the
# reshape.
#
# We can probably safely assume (for now) that `optic_vec(d.dist)` doesn't return anything
# that has a more complicated structure than an array index. For example, `d.dist` couldn't
# be something like LKJCholesky because you can't call `reshape(LKJCholesky(...), ...)`
# anyway.
function Plaice.optic_vec(d::D.ReshapedDistribution)
    original_optics = Plaice.optic_vec(d.dist)
    linear_indices_original = LinearIndices(size(d.dist))
    cartesian_indices_reshaped = CartesianIndices(size(d))
    mapped_optics = map(original_optics) do opt
        if opt isa VarNames.Index
            # Don't know how to generally handle this yet. Probably not an issue yet
            # because Distributions.jl is not fancy enough to have complicated axes.
            if !isempty(opt.kw)
                error("optic_vec for ReshapedDistribution only supports simple Index optics")
            end
            # Map the indices through the reshape
            linear_index = linear_indices_original[opt.ix...]
            new_cartesian_index = cartesian_indices_reshaped[linear_index]
            return VarNames.Index(new_cartesian_index.I, (;), opt.child)
        else
            error("optic_vec for ReshapedDistribution only supports Index optics")
        end
    end
    return mapped_optics
end
# If `d.dist` is univariate that is a special case because `optic_vec(d.dist)` would return
# [Iden]`. In that case we need to tack on the array indices.
function Plaice.optic_vec(d::ReshapedUnivariateDistribution)
    # size(d) should be a tuple that contains only 1's, so we can just reuse it
    return [VarNames.Index(size(d), (;))]
end

# unconstrained_optic_vec is the same...
function Plaice.unconstrained_optic_vec(d::D.ReshapedDistribution)
    original_optics = Plaice.unconstrained_optic_vec(d.dist)
    linear_indices_original = LinearIndices(size(d.dist))
    cartesian_indices_reshaped = CartesianIndices(size(d))
    mapped_optics = map(original_optics) do opt
        if opt isa VarNames.Index
            if !isempty(opt.kw)
                error(
                    "unconstrained_optic_vec for ReshapedDistribution only supports simple Index optics",
                )
            end
            linear_index = linear_indices_original[opt.ix...]
            new_cartesian_index = cartesian_indices_reshaped[linear_index]
            return VarNames.Index(new_cartesian_index.I, (;), opt.child)
        elseif opt === nothing
            # ... but we just need to make sure to forward any `nothing`s.
            return nothing
        else
            error("unconstrained_optic_vec for ReshapedDistribution only supports Index optics")
        end
    end
    return mapped_optics
end
function Plaice.unconstrained_optic_vec(d::ReshapedUnivariateDistribution)
    return [VarNames.Index(size(d), (;))]
end

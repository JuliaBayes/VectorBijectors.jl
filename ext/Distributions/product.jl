@generated function Plaice._make_transform_inner(
    dists::NTuple{NDists,D.Distribution},
    indiv_transform_fn,
    length_fn,
    struct_type,
) where {NDists}
    exprs = []
    trfms = Expr(:tuple)
    for i in 1:NDists
        push!(trfms.args, :(indiv_transform_fn(dists[$i])))
    end
    push!(exprs, :(trfms = $trfms))
    push!(exprs, :(ranges = ()))
    push!(exprs, :(offset = 1))
    for i in 1:NDists
        push!(exprs, :(this_length = length_fn(dists[$i])))
        push!(exprs, :(ranges = (ranges..., offset:(offset+this_length-1))))
        push!(exprs, :(offset += this_length))
    end
    push!(exprs, :(return struct_type(trfms, ranges, size(dists[1]))))
    return Expr(:block, exprs...)
end

function Plaice._make_transform_inner(
    dists::AbstractArray{<:D.Distribution},
    indiv_transform_fn,
    length_fn,
    struct_type,
)
    # map(indiv_transform_fn, dists) causes some Enzyme errors when used with DPPL
    # https://github.com/TuringLang/DynamicPPL.jl/issues/1304
    trfms = indiv_transform_fn.(dists)
    ranges = Array{UnitRange{Int}}(undef, size(dists)...)
    offset = 1
    for (i, dist) in enumerate(dists)
        this_length = length_fn(dist)
        ranges[i] = offset:(offset+this_length-1)
        offset += this_length
    end
    return struct_type(trfms, ranges, size(dists[1]))
end

for (product_type, dist_field) in (
    (D.ProductNamedTupleDistribution, :dists),
    (D.ProductDistribution, :dists),
    # Annoyingly, vectors of univariate distributions become D.Product rather than
    # D.ProductDistribution (which handles all other tuple/arrays).
    (D.Product, :v),
)
    @eval begin
        function Plaice.from_vec(d::$product_type)
            return Plaice._make_transform(
                d.$dist_field,
                Plaice.from_vec,
                Plaice.vec_length,
                Plaice.ProductVecInvTransform,
            )
        end
        function Plaice.from_unconstrained_vec(d::$product_type)
            return Plaice._make_transform(
                d.$dist_field,
                Plaice.from_unconstrained_vec,
                Plaice.unconstrained_vec_length,
                Plaice.ProductVecInvTransform,
            )
        end
        function Plaice.to_vec(d::$product_type)
            return Plaice._make_transform(
                d.$dist_field,
                Plaice.to_vec,
                Plaice.vec_length,
                Plaice.ProductVecTransform,
            )
        end
        function Plaice.to_unconstrained_vec(d::$product_type)
            return Plaice._make_transform(
                d.$dist_field,
                Plaice.to_unconstrained_vec,
                Plaice.unconstrained_vec_length,
                Plaice.ProductVecTransform,
            )
        end

        Plaice.vec_length(d::$product_type) = sum(Plaice.vec_length, d.$dist_field)
        Plaice.unconstrained_vec_length(d::$product_type) =
            sum(Plaice.unconstrained_vec_length, d.$dist_field)
    end
end

for f in (:optic_vec, :unconstrained_optic_vec)
    for (product_type, dist_field) in ((D.Product, :v), (D.ProductDistribution, :dists))
        @eval begin
            function Plaice.$f(d::$product_type)
                optics = Union{}[]
                idxs = Plaice._cartesian_indices(d.$dist_field)
                for (idx, dist) in zip(idxs, d.$dist_field)
                    this_dist_optics = Plaice.$f(dist)
                    new_optics = map(optic -> Plaice.append_index(optic, idx), this_dist_optics)
                    optics = vcat(optics, new_optics)
                end
                return optics
            end
        end
    end

    @eval begin
        function Plaice.$f(d::D.ProductNamedTupleDistribution)
            optics = Union{}[]
            for (nm, dist) in pairs(d.dists)
                this_dist_optics = Plaice.$f(dist)
                new_optics = map(optic -> Plaice.prepend_symbol(nm, optic), this_dist_optics)
                optics = vcat(optics, new_optics)
            end
            return optics
        end
    end
end

Plaice.has_constant_vec_bijector(::Type{<:IDENTITY_UNIVARIATES}) = true
Plaice.has_constant_vec_bijector(::Type{<:POSITIVE_UNIVARIATES}) = true
# between 0 and 1
function Plaice.has_constant_vec_bijector(
    ::Type{<:Union{D.Beta,D.KSOneSided,D.NoncentralBeta,D.LogitNormal}},
)
    return true
end
Plaice.has_constant_vec_bijector(::Type{<:D.DiscreteUnivariateDistribution}) = true
# Multivariates
Plaice.has_constant_vec_bijector(::Type{<:D.AbstractMvNormal}) = true
Plaice.has_constant_vec_bijector(::Type{<:D.AbstractMvTDist}) = true
Plaice.has_constant_vec_bijector(::Type{<:D.AbstractMvLogNormal}) = true
Plaice.has_constant_vec_bijector(::Type{<:SIMPLEX_MULTIVARIATES}) = true
Plaice.has_constant_vec_bijector(::Type{<:D.DiscreteMultivariateDistribution}) = true

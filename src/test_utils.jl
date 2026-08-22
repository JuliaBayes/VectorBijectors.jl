using Test
using LinearAlgebra: logabsdet, Cholesky, UpperTriangular, LowerTriangular
import DifferentiationInterface as DI
import EnzymeCore as EC

# Would like to use FiniteDifferences, but very easy to run into issues with
# https://juliadiff.org/FiniteDifferences.jl/latest/#Dealing-with-Singularities
const ref_adtype = DI.AutoForwardDiff()

const default_adtypes = [
    DI.AutoReverseDiff(),
    DI.AutoReverseDiff(; compile=true),
    DI.AutoMooncake(),
    DI.AutoMooncakeForward(),
    DI.AutoEnzyme(; mode=EC.Forward, function_annotation=EC.Const),
    DI.AutoEnzyme(; mode=EC.Reverse, function_annotation=EC.Const),
]

# Check if a distribution is continuous.
function is_continuous end

# Overload in order to pretty-print distributions. Otherwise things like MvNormal can get
# super ugly.
test_name(d) = repr(d)

# AD will give nonsense results at the limits of censored distributions (since the gradient
# is not well-defined), so we avoid generating samples that are exactly at the limits.
function rand_safe_ad end

# isapprox is not defined for some samples (specifically Cholesky and NTs), so we need to
# patch that
function _isapprox_safe(x, y; kwargs...)
    return isapprox(x, y; kwargs...)
end
function _isapprox_safe(x::NamedTuple{names}, y::NamedTuple{names}; kwargs...) where {names}
    for name in names
        if !_isapprox_safe(x[name], y[name]; kwargs...)
            return false
        end
    end
    return true
end
function _isapprox_safe(x::Cholesky, y::Cholesky; kwargs...)
    if x.uplo != y.uplo || size(x.UL) != size(y.UL)
        return false
    end
    return isapprox(x.UL, y.UL; kwargs...)
end

# When testing logjac for distributions where `vec_length(d) != unconstrained_vec_length(d)`, if
# we naively try to compute the logjacobian of the transformation from vector to unconstrained
# vector form (or vice versa), it will error because the dimensions don't match (i.e., the
# Jacobian is not square). See
# https://turinglang.org/Bijectors.jl/stable/defining_examples/#Stereographic-projection
# for an example of how to work around this issue.
#
# Here we define a function which converts a sample from `d` to a vector of length
# `unconstrained_vec_length(d)` but does NOT perform linking. This allows us to compute the
# Jacobian from `to_vec_for_logjac_test(d)(x)` to `to_unconstrained_vec(d)(x)`, which will be
# square. The fallback definition is just to_vec(d), but we can overload this for specific
# distributions.
to_vec_for_logjac_test(d) = to_vec(d)
from_vec_for_logjac_test(d) = from_vec(d)

function test_all(
    d;
    expected_zero_allocs=(),
    adtypes=default_adtypes,
    ad_atol=1e-10,
    ad_rtol=sqrt(eps()),
    roundtrip_atol=1e-10,
    roundtrip_rtol=sqrt(eps()),
    test_in_support=is_continuous(d),
    test_construction_type_stable=true,
)
    @info "Testing $(test_name(d))"

    reactant_loaded = any(
        k -> k.uuid == Base.UUID("3c362404-f566-11ee-1572-e11a4b42c853"),
        keys(Base.loaded_modules),
    )

    @testset "$(test_name(d))" begin
        test_bijector_interface(d)
        test_roundtrip(d)
        test_roundtrip_inverse(d, test_in_support, roundtrip_atol, roundtrip_rtol)
        test_type_stability(d, test_construction_type_stable)
        test_vec_lengths(d)
        test_optics(d)
        test_allocations(d, expected_zero_allocs)
        test_logjac(d, ad_atol, ad_rtol)
        test_ad(d, adtypes, ad_atol, ad_rtol)
        if reactant_loaded
            test_reactant(d)
        end
    end
end

"""
Sanity check: make sure that the bijectors exist and are each other's inverses.
"""
function test_bijector_interface(d)
    @testset "bijector interface: $(test_name(d))" begin
        fv = from_vec(d)
        tv = to_vec(d)
        @test fv isa AbstractBijector
        @test tv isa AbstractBijector
        @test inverse(fv) == tv
        @test inverse(tv) == fv
        fuv = from_unconstrained_vec(d)
        tuv = to_unconstrained_vec(d)
        @test fuv isa AbstractBijector
        @test tuv isa AbstractBijector
        @test inverse(fuv) == tuv
        @test inverse(tuv) == fuv
    end
end

"""
Test that from_vec and to_vec are inverses, and likewise for from_unconstrained_vec and
to_unconstrained_vec. This test checks `x ≈ from_vec(d)(to_vec(d)(x))` for random samples `x ~ d`
(and likewise for the unconstrained transforms).
"""
function test_roundtrip(d)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip: $(test_name(d))" begin
        for _ in 1:1000
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                frvs = from_vec(d)
                @test _isapprox_safe(x, frvs(ffwd(x)))
            end
        end
    end
    @testset "roundtrip (unconstrained): $(test_name(d))" begin
        for _ in 1:1000
            @testset let x = rand(d), d = d
                ffwd = to_unconstrained_vec(d)
                frvs = from_unconstrained_vec(d)
                xnew = frvs(ffwd(x))
                # https://github.com/TuringLang/Bijectors.jl/issues/441
                if _is_joint_order_statistics(d) &&
                   (any(isnan, xnew) || !all(isfinite, xnew))
                    @warn "NaNs or Inf produced in roundtrip test for $(test_name(d)), skipping isapprox test"
                else
                    @test _isapprox_safe(x, xnew)
                end
            end
        end
    end
end

_is_joint_order_statistics(::Any) = false

# The implementation of this requires Distributions.jl so has to go in the ext
can_test_in_support(d, x) = false
test_in_support(d, x) = error("not implemented")

"""
Test that from_unconstrained_vec and to_unconstrained_vec are inverses.

If `test_in_support_flag`, then this additionally also tests that `from_unconstrained_vec(dist)`
actually does map random vectors to the support of the distribution (i.e., `finv(y)` for
some random `y` is in the support of `d`).

If the distribution is not continuous, we can't really check this (in fact the test is quite
meaningless). So for discrete distributions this test is skipped. There are also other
occasions where we disable this because of e.g. numerical issues, like for LKJ.
"""
function test_roundtrip_inverse(d, test_in_support_flag, atol, rtol)
    # TODO: Use smarter test generation e.g. with property-based testing or at least
    # generate random parameters across the support
    @testset "roundtrip inverse (unconstrained): $(test_name(d))" begin
        len = unconstrained_vec_length(d)
        x = rand(d)

        if test_in_support_flag && !can_test_in_support(d, x)
            @warn "Skipping test_in_support for $(test_name(d)) because it is not implemented"
            test_in_support_flag = false
        end

        for _ in 1:100
            @testset let y = randn(len), d = d
                ffwd = to_unconstrained_vec(d)
                frvs = from_unconstrained_vec(d)
                x = frvs(y)
                if test_in_support_flag
                    test_in_support(d, x)
                end

                ynew = ffwd(x)
                if _is_joint_order_statistics(d) && (
                    any(isnan, x) ||
                    !all(isfinite, x) ||
                    any(isnan, ynew) ||
                    !all(isfinite, ynew)
                )
                    @warn "NaNs or Inf produced in roundtrip test for $(test_name(d)), skipping isapprox test"
                else
                    @test _isapprox_safe(y, ynew; atol=atol, rtol=rtol)
                end
            end
        end
    end
end

"""
Test that the conversions to and from vector and unconstrained vector forms for the given
distribution `d` are type-stable.

Sometimes the *creation* of the bijector itself is not type-stable (e.g. for product
distributions with heterogeneous components), but once the bijector is created, the
conversions should be type stable. To disable type stability checks for the construction,
set `test_construction_type_stable=false`.
"""
function test_type_stability(d, test_construction_type_stable=true)
    x = rand(d)
    @testset "type stability: $(test_name(d))" begin
        @testset let x = x, d = d
            if test_construction_type_stable
                @inferred to_vec(d)
                @inferred from_vec(d)
            end
            ffwd = to_vec(d)
            frvs = from_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
    @testset "type stability (unconstrained): $(test_name(d))" begin
        @testset let x = x, d = d
            if test_construction_type_stable
                @inferred to_unconstrained_vec(d)
                @inferred from_unconstrained_vec(d)
            end
            ffwd = to_unconstrained_vec(d)
            frvs = from_unconstrained_vec(d)
            @inferred ffwd(x)
            y = ffwd(x)
            @inferred frvs(y)
        end
    end
end

"""
Test that the optics produced by `optic_vec` for the given distribution `d` line up with the
values produced by `to_vec`.
"""
function test_optics(d)
    @testset "optic_vec: $(test_name(d))" begin
        o = optic_vec(d)
        x = rand(d)
        v = to_vec(d)(x)
        for (optic, value) in zip(o, v)
            if optic !== nothing
                @test optic(x) == value
            end
        end
    end

    @testset "unconstrained_optic_vec: $(test_name(d))" begin
        # This is a lot harder to test. What we need to prove is that
        #       x = rand(d)
        #       lv = to_unconstrained_vec(d)(x)
        #       lo = unconstrained_optic_vec(d)
        #  then for each element of `lv, lo` we have that `lv[i]` depends only on lo[i](x)
        #  and not any other elements of `x`. Conceptually, this means that if we take the
        #  Jacobian of the link transform, row `i` should have nonzeros only in the columns
        #  corresponding to `lo[i]`. This is a bit finicky to do because `x` might not be a
        #  vector(!) so we need to flatten everything first, using `to_vec`.
        x = rand(d)
        xvec = to_vec(d)(x)
        yvec = to_unconstrained_vec(d)(x)
        J = DI.jacobian(to_unconstrained_vec(d) ∘ from_vec(d), ref_adtype, xvec)
        o = optic_vec(d)
        lo = unconstrained_optic_vec(d)
        for i in 1:length(yvec)
            unconstrained_optic = lo[i]
            if unconstrained_optic !== nothing
                # If the optic is non-nothing, then it refers to a specific element
                # of x. That means that we should be able to find, which index of the
                # input `xvec` it corresponds to, by finding the index `j` where
                # `optic_vec(d)[j] === unconstrained_optic`.
                nonzero_index = findfirst(j -> o[j] == unconstrained_optic, 1:length(xvec))
                if nonzero_index === nothing
                    error(
                        "unconstrained_optic_vec produced an optic not found in optic_vec",
                    )
                end
                for j in 1:length(xvec)
                    if j != nonzero_index
                        @test iszero(J[i, j])
                    end
                end
            end
        end
    end
end

"""
Test that the lengths of the vectors produced by the conversions to vector and unconstrained
vector forms for the given distribution `d` match those reported by `vec_length` and
`unconstrained_vec_length`.
"""
function test_vec_lengths(d)
    @testset "vector lengths: $(test_name(d))" begin
        for _ in 1:10
            @testset let x = rand(d), d = d
                y = to_vec(d)(x)
                @test length(y) == vec_length(d)
            end
        end
    end
    @testset "vector lengths (unconstrained): $(test_name(d))" begin
        for _ in 1:10
            @testset let x = rand(d), d = d
                y = to_unconstrained_vec(d)(x)
                @test length(y) == unconstrained_vec_length(d)
            end
        end
    end
end

"""
Test that the conversions to and from vector and unconstrained vector forms for the given
distribution `d` do not cause any heap allocations for the functions specified in
`expected_zero_allocs`.
"""
function test_allocations(d, expected_zero_allocs=())
    ALLOWED_FUNCTIONS = (to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec)
    if any(f -> !(f in ALLOWED_FUNCTIONS), expected_zero_allocs)
        throw(ArgumentError("expected_zero_allocs can only contain: $ALLOWED_FUNCTIONS"))
    end
    # For univariates, to_vec and to_unconstrained_vec always cause allocations because they have
    # to create a new vector.
    # TODO: Generalise to multivariates etc
    x = rand(d)
    @testset "allocations: $(test_name(d))" begin
        @testset let x = x, d = d
            if to_vec in expected_zero_allocs
                ffwd = to_vec(d)
                ffwd(x)
                @test (@allocations ffwd(x)) == 0
            end
            if from_vec in expected_zero_allocs
                yvec = to_vec(d)(x)
                frvs = from_vec(d)
                frvs(yvec)
                @test (@allocations frvs(yvec)) == 0
            end
        end
    end
    @testset "allocations (unconstrained): $(test_name(d))" begin
        @testset let x = x, d = d
            if to_unconstrained_vec in expected_zero_allocs
                ffwd = to_unconstrained_vec(d)
                ffwd(x)
                @test (@allocations ffwd(x)) == 0
            end
            if from_unconstrained_vec in expected_zero_allocs
                yvec = to_unconstrained_vec(d)(x)
                frvs = from_unconstrained_vec(d)
                frvs(yvec)
                @test (@allocations frvs(yvec)) == 0
            end
        end
    end
end

"""
Test that the analytical log-Jacobians provided in this package are correct by comparing
against AD-calculated log-Jacobians for the given distribution `d`.
"""
function test_logjac(d, atol, rtol)
    # Vectorisation logjacs should be zero because they are just reshapes.
    @testset "logjac: $(test_name(d))" begin
        for _ in 1:100
            @testset let x = rand(d), d = d
                ffwd = to_vec(d)
                y, logjac = with_logabsdet_jacobian(ffwd, x)
                @test _isapprox_safe(y, ffwd(x); atol=atol, rtol=rtol)
                @test iszero(logjac)
                frvs = from_vec(d)
                x_recon, logjac = with_logabsdet_jacobian(frvs, y)
                @test _isapprox_safe(x_recon, frvs(y); atol=atol, rtol=rtol)
                @test iszero(logjac)
            end
        end
    end

    # Link logjacs will not be zero, so we need to check against a chosen backend. Because
    # Jacobians need to map from vector to vector, here we test the transformation of the
    # vectorised form to the unconstrained vectorised form via the original sample.
    @testset "logjac (unconstrained): $(test_name(d))" begin
        for _ in 1:100
            x = rand_safe_ad(d)

            @testset let x = x, d = d
                # As a sanity check we should make sure that to_vec_for_logjac_test and
                # from_vec_for_logjac_test are inverses. If they aren't, then that brings
                # the entire testset into question!
                @test _isapprox_safe(
                    x,
                    from_vec_for_logjac_test(d)(to_vec_for_logjac_test(d)(x));
                    atol=atol,
                    rtol=rtol,
                )
            end

            @testset let x = x, d = d
                # Forward
                xvec = to_vec(d)(x)
                ffwd = to_unconstrained_vec(d) ∘ from_vec(d)
                y, vbt_logjac = with_logabsdet_jacobian(ffwd, xvec)
                @test _isapprox_safe(y, ffwd(xvec); atol=atol, rtol=rtol)
                # For the AD calculation we need to use to/from_vec_for_logjac_test instead,
                # to make sure that the Jacobian is square.
                ad_xvec = to_vec_for_logjac_test(d)(x)
                ad_ffwd = to_unconstrained_vec(d) ∘ from_vec_for_logjac_test(d)
                ad_logjac = first(logabsdet(DI.jacobian(ad_ffwd, ref_adtype, ad_xvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol rtol = rtol
            end

            @testset let x = x, d = d
                # Reverse
                yvec = to_unconstrained_vec(d)(x)
                vbt_frvs = to_vec(d) ∘ from_unconstrained_vec(d)
                x, vbt_logjac = with_logabsdet_jacobian(vbt_frvs, yvec)
                @test _isapprox_safe(x, vbt_frvs(yvec); atol=atol, rtol=rtol)
                # For the AD calculation we need to use to/from_vec_for_logjac_test instead,
                # to make sure that the Jacobian is square.
                ad_frvs = to_vec_for_logjac_test(d) ∘ from_unconstrained_vec(d)
                ad_logjac = first(logabsdet(DI.jacobian(ad_frvs, ref_adtype, yvec)))
                @test vbt_logjac ≈ ad_logjac atol = atol rtol = rtol
            end
        end
    end
end

"""
Test that various AD backends can differentiate the conversions to and from vector and
unconstrained vector forms for the given distribution `d`.
"""
function test_ad(d, adtypes::Vector{<:DI.AbstractADType}, atol, rtol)
    # If `d` is a discrete distribution, Mooncake refuses to differentiate through the
    # transforms (which are just identity transforms). Likewise, Enzyme will throw an
    # error saying that the output is Const but was not marked as such.
    #
    # Arguably, the other AD backends probably should also refuse to differentiate it, but
    # they do actually return the right 'gradients' so we can test them.
    adtypes = if !is_continuous(d)
        filter(adtypes) do adtype
            !(
                adtype isa DI.AutoMooncake ||
                adtype isa DI.AutoMooncakeForward ||
                adtype isa DI.AutoEnzyme
            )
        end
    else
        adtypes
    end

    @testset "AD forward: $(test_name(d))" begin
        x = rand_safe_ad(d)
        xvec = to_vec(d)(x)
        ffwd = to_unconstrained_vec(d) ∘ from_vec(d)
        ref_jac = DI.jacobian(ffwd, ref_adtype, xvec)

        ladj(xvec) = last(with_logabsdet_jacobian(ffwd, xvec))
        ref_grad_ladj = DI.gradient(ladj, ref_adtype, xvec)

        for adtype in adtypes
            # Check that the forward transform is differentiable
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(ffwd, adtype, xvec)
                @test ref_jac ≈ ad_jac atol = atol rtol = rtol
            end
            # Check that the logjac calculation itself is differentiable
            @testset let x = x, adtype = adtype, d = d
                ad_grad_ladj = DI.gradient(ladj, adtype, xvec)
                @test ref_grad_ladj ≈ ad_grad_ladj atol = atol rtol = rtol
            end
        end
    end

    @testset "AD reverse: $(test_name(d))" begin
        x = rand_safe_ad(d)
        yvec = to_unconstrained_vec(d)(x)
        frvs = to_vec(d) ∘ from_unconstrained_vec(d)
        ref_jac = DI.jacobian(frvs, ref_adtype, yvec)

        ladj(yvec) = last(with_logabsdet_jacobian(frvs, yvec))
        ref_grad_ladj = DI.gradient(ladj, ref_adtype, yvec)

        for adtype in adtypes
            # Check that the reverse transform is differentiable
            @testset let x = x, adtype = adtype, d = d
                ad_jac = DI.jacobian(frvs, adtype, yvec)
                @test ref_jac ≈ ad_jac atol = atol rtol = rtol
            end
            # Check that the logjac calculation itself is differentiable
            @testset let x = x, adtype = adtype, d = d
                ad_grad_ladj = DI.gradient(ladj, adtype, yvec)
                @test ref_grad_ladj ≈ ad_grad_ladj atol = atol rtol = rtol
            end
        end
    end
end

# Overloaded in ReactantExt
function test_reactant end

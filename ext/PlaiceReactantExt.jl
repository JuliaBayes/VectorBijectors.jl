module PlaiceReactantExt

using Plaice
using Enzyme
using Reactant
using Test
import DifferentiationInterface as DI

function Plaice.test_reactant(d)
    x = Plaice.rand_safe_ad(d)

    xvec = Plaice.to_vec(d)(x)
    yvec = Plaice.to_unconstrained_vec(d)(x)

    ffwd = Plaice.to_unconstrained_vec(d) ∘ Plaice.from_vec(d)
    frvs = Plaice.to_vec(d) ∘ Plaice.from_unconstrained_vec(d)

    xvec_r = Reactant.to_rarray(xvec)
    yvec_r = Reactant.to_rarray(yvec)

    @testset "Reactant: $(Plaice.test_name(d))" begin
        @testset "correctness: with_logabsdet_jacobian forward" begin
            expected_y, expected_ladj = Plaice.with_logabsdet_jacobian(ffwd, xvec)
            result = @allowscalar @jit ffwd(xvec_r)
            @test Array(result) ≈ expected_y
            result_and_lj = @allowscalar @jit Plaice.with_logabsdet_jacobian(ffwd, xvec_r)
            @test Array(result_and_lj[1]) ≈ expected_y
            @test Float64(result_and_lj[2]) ≈ expected_ladj
        end

        @testset "correctness: with_logabsdet_jacobian reverse" begin
            expected_x, expected_ladj = Plaice.with_logabsdet_jacobian(frvs, yvec)
            result = @allowscalar @jit frvs(yvec_r)
            @test Array(result) ≈ expected_x
            result_and_lj = @allowscalar @jit Plaice.with_logabsdet_jacobian(frvs, yvec_r)
            @test Array(result_and_lj[1]) ≈ expected_x
            @test Float64(result_and_lj[2]) ≈ expected_ladj
        end

        @testset "gradient: forward transform" begin
            # Have to use sum to collapse to scalar since jacobian doesn't work inside
            # Reactant. https://github.com/EnzymeAD/Reactant.jl/pull/2839
            sum_fwd(v) = sum(ffwd(v))
            ref_grad = DI.gradient(sum_fwd, Plaice.ref_adtype, xvec)
            result = @allowscalar @jit Enzyme.gradient(Enzyme.Reverse, sum_fwd, xvec_r)
            @test Array(only(result)) ≈ ref_grad
        end

        @testset "gradient: reverse transform" begin
            # Have to use sum to collapse to scalar since jacobian doesn't work inside
            # Reactant. https://github.com/EnzymeAD/Reactant.jl/pull/2839
            sum_rvs(v) = sum(frvs(v))
            ref_grad = DI.gradient(sum_rvs, Plaice.ref_adtype, yvec)
            result = @allowscalar @jit Enzyme.gradient(Enzyme.Reverse, sum_rvs, yvec_r)
            @test Array(only(result)) ≈ ref_grad
        end

        @testset "gradient: forward ladj" begin
            ladj_fwd(v) = last(Plaice.with_logabsdet_jacobian(ffwd, v))
            ref_grad = DI.gradient(ladj_fwd, Plaice.ref_adtype, xvec)
            result = @allowscalar @jit Enzyme.gradient(Enzyme.Reverse, ladj_fwd, xvec_r)
            # result is a 1-elem tuple containing the Reactant array
            @test Array(only(result)) ≈ ref_grad
        end

        @testset "gradient: reverse ladj" begin
            ladj_rvs(v) = last(Plaice.with_logabsdet_jacobian(frvs, v))
            ref_grad = DI.gradient(ladj_rvs, Plaice.ref_adtype, yvec)
            result = @allowscalar @jit Enzyme.gradient(Enzyme.Reverse, ladj_rvs, yvec_r)
            # result is a 1-elem tuple containing the Reactant array
            @test Array(only(result)) ≈ ref_grad
        end
    end
end

end # module PlaiceReactantExt

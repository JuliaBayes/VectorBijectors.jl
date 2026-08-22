module PlaicePMMultivariateTests

using ProbabilityMeasures
using LinearAlgebra
using Test
using Plaice
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Mooncake: Mooncake

const PM = ProbabilityMeasures

multivariates = [
    PM.MvNormal([0.0, 0.0], I),
    PM.MvNormal(
        [1.0, 2.0, 3.0], [2.0 0.0 0.0; -1.0 2.0 0.0; 0.5 -0.5 1.5]
    ),
]

@testset "PM Multivariates" begin
    for d in multivariates
        Plaice.test_all(
            d;
            expected_zero_allocs=(
                to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec
            ),
        )
    end
end

end # module PlaicePMMultivariateTests

using Test: @test

function Plaice.is_continuous(
    ::PM.AbstractProbabilityMeasure{<:Any,VS},
) where {VS<:PM.ValueSupport}
    return VS <: PM.Continuous
end
Plaice.test_name(d::PM.AbstractProbabilityMeasure) = repr(d)
Plaice.rand_safe_ad(d::PM.AbstractProbabilityMeasure) = rand(d)

Plaice.can_test_in_support(d::PM.AbstractProbabilityMeasure, x) = true
function Plaice.test_in_support(d::PM.AbstractProbabilityMeasure, x)
    @test PM.insupport(d, x)
end

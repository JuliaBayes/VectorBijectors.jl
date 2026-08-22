module PlaiceProbabilityMeasuresExt

using Plaice
using ProbabilityMeasures
using VarNames

const PM = ProbabilityMeasures

include("ProbabilityMeasures/univariates.jl")
include("ProbabilityMeasures/multivariates.jl")
include("ProbabilityMeasures/test_utils.jl")

end # module PlaiceProbabilityMeasuresExt

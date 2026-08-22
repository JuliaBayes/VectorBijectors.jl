module PlaiceTests

using Plaice
using Test

# gives loads of errors on Windows CI
if !Sys.iswindows()
    import Pkg
    Pkg.add("Reactant")
end

@testset "Plaice.jl" begin
    include("interface.jl")
    include("distributions/univariate.jl")
    include("distributions/multivariate.jl")
    include("distributions/matrix.jl")
    include("distributions/reshaped.jl")
    include("distributions/cholesky.jl")
    include("distributions/order.jl")
    include("distributions/product.jl")
end

end # module PlaiceTests

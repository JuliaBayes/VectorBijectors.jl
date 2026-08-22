module Plaice

using VarNames: VarNames, @opticof
import ChangesOfVariables: with_logabsdet_jacobian
import InverseFunctions: inverse

include("common.jl")
include("bijector_types.jl")
include("interface.jl")
export from_vec
export to_vec
export from_unconstrained_vec
export to_unconstrained_vec
export vec_length
export unconstrained_vec_length
export optic_vec
export unconstrained_optic_vec
# utils
export has_constant_vec_bijector
export scalar_to_scalar_bijector
export is_monotonically_increasing
export is_monotonically_decreasing
export AbstractBijector
export TypedIdentity, Log, Untruncate
# re-exports
export with_logabsdet_jacobian
export logabsdet_jacobian
export inverse

include("univariate/univariate.jl")
include("univariate/positive.jl")
include("univariate/truncated.jl")

include("multivariate/mvlognormal.jl")

include("matrix/matrix.jl")
include("matrix/normal.jl")
include("matrix/posdef.jl")
include("matrix/lkj.jl")

include("order/order.jl")
include("reshaped/reshaped.jl")
include("cholesky/cholesky.jl")

include("product/product.jl")
include("product/fill.jl")

# Put last to avoid cluttering namespace
include("test_utils.jl")

end # module Plaice

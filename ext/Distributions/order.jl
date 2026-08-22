# OrderStatistic can only ever wrap univariate distributions so these can just delegate to
# the underlying distribution.
Plaice.to_vec(d::D.OrderStatistic) = Plaice.to_vec(d.dist)
Plaice.from_vec(d::D.OrderStatistic) = Plaice.from_vec(d.dist)
Plaice.to_unconstrained_vec(d::D.OrderStatistic) = Plaice.to_unconstrained_vec(d.dist)
Plaice.from_unconstrained_vec(d::D.OrderStatistic) = Plaice.from_unconstrained_vec(d.dist)
# We don't need to implement the other methods as OrderStatistic is a subtype of
# UnivariateDistribution, so we can just use the default methods.

# Here, because `d.dist` isa UnivariateDistribution, we can get its scalar-to-scalar
# bijector and then rewrap that inner bijector into a JointOrderWrap to get the desired
# behavior.
Plaice.to_vec(::D.JointOrderStatistics) = Plaice.TypedIdentity()
function Plaice.to_unconstrained_vec(d::D.JointOrderStatistics)
    return Plaice.JointOrderWrap(Plaice.scalar_to_scalar_bijector(d.dist))
end
Plaice.from_vec(::D.JointOrderStatistics) = Plaice.TypedIdentity()
function Plaice.from_unconstrained_vec(d::D.JointOrderStatistics)
    return Plaice.InverseJointOrderWrap(Plaice.inverse(Plaice.scalar_to_scalar_bijector(d.dist)))
end
# Since D.JointOrderStatistics is a subtype of MultivariateDistribution, we can use the
# default definitions for vec_length and optic_vec.
Plaice.unconstrained_vec_length(d::D.JointOrderStatistics) = Plaice.vec_length(d)
# TODO: Technically, the first element can be @opticof(_[1]) so this is not technically
# correct.
Plaice.unconstrained_optic_vec(d::D.JointOrderStatistics) = fill(nothing, Plaice.vec_length(d))

Plaice._is_joint_order_statistics(::D.JointOrderStatistics) = true

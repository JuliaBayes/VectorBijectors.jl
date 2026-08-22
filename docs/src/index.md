# [Plaice](@id vector)

Plaice.jl converts samples from distributions to and from **vectors**.

It assumes that there are three forms of samples from a distribution `d` that we are interested in:

 1. **The original form**, which is what `rand(d)` returns.

 2. **A vectorised form**, which is a vector that contains a flattened version of the original form.

 3. **A unconstrained vectorised form**, which is a vector in which:

      + each element is independent; and
      + each element is unconstrained (can take any value in ℝ).

Note that because of the independence requirement, the unconstrained vectorised form may have a different dimension to the vectorised form.
For example, when sampling from a `Dirichlet` distribution, the original form is a vector that always sums to 1.
The unconstrained vectorised form will have one element less than the original form, because this constraint is eliminated.

Plaice.jl provides functionality to convert between these three forms, via subtypes of `Plaice.AbstractBijector`.
Assuming that `x = rand(d)` for some distribution `d`:

  - `to_vec(d)` is an `AbstractBijector` which converts `x` to the vectorised form
  - `from_vec(d)` is the inverse of `to_vec(d)`, also an `AbstractBijector`
  - `vec_length(d)` returns the length of `to_vec(d)(x)`
  - `optic_vec(d)` returns a vector of optics that describes how each element of `to_vec(d)(x)` is accessed from `x`
  - `to_unconstrained_vec(d)` is an `AbstractBijector` which converts `x` to the unconstrained vectorised form
  - `from_unconstrained_vec(d)` is the inverse of `to_unconstrained_vec(d)`, also an `AbstractBijector`
  - `unconstrained_vec_length(d)` returns the length of `to_unconstrained_vec(d)(x)`
  - `unconstrained_optic_vec(d)` returns a vector of optics that describes how each element of `to_unconstrained_vec(d)(x)` is accessed from `x` (if possible)

For example:

```julia
julia> using Plaice, Distributions

julia> d = Beta(2, 2);
       x = rand(d);  # x is between 0 and 1
0.5602086057097567

julia> to_vec(d)(x)
1-element Vector{Float64}:
 0.5602086057097567

julia> to_unconstrained_vec(d)(x)
1-element Vector{Float64}:
 0.24200871395677753
```

Subtypes of `AbstractBijector` must minimally implement `ChangesOfVariables.with_logabsdet_jacobian` as well as `InverseFunctions.inverse`.
This is tested for all of Plaice's own bijectors.
By defining `with_logabsdet_jacobian`, you can also get default definitions for `(b::AbstractBijector)(x)` and `logabsdet_jacobian(b, x)`.

```@docs
logabsdet_jacobian
```

## Implementing your own vector bijector

The full Plaice interface consists of the following functions:

```@docs
Plaice.AbstractBijector
Plaice.from_vec
Plaice.to_vec
Plaice.from_unconstrained_vec
Plaice.to_unconstrained_vec
Plaice.vec_length
Plaice.unconstrained_vec_length
Plaice.optic_vec
Plaice.unconstrained_optic_vec
```

In practice, if your distribution is a univariate distribution, you will probably only need to implement `scalar_to_scalar_bijector` (see below).

For multivariate and matrix distributions, there are default implementations of the non-unconstrained versions (i.e., `from_vec`, `to_vec`, `vec_length`, and `optic_vec`) which should already be optimal.
However you will have to define the unconstrained versions (see [the examples page](@ref example) for more info).

If you have a very customised distribution, you will likely have to implement all the functions yourself.

## Known constant bijectors

Additionally, if your distribution is likely to be used in part of a product distribution, it can lead to substantial performance improvements to overload `has_constant_vec_bijector` to return `true` (but make sure to only do this if the bijector is genuinely constant!):

```@docs
Plaice.has_constant_vec_bijector
```

## Univariate distributions

For univariate distributions the default definition is to generate a bijector that inspects the minimum and maximum of the distribution.
While this will work correctly, it might not be the most performant.
You can manually define the Plaice API for univariate distributions, but it is probably faster to just overload the single function `scalar_to_scalar_bijector`: everything else will be automatically dervied.

```@docs
Plaice.scalar_to_scalar_bijector
```

Univariate distributions tend to fall into one of the following categories:

```@docs
Plaice.TypedIdentity
Plaice.Log
Plaice.Untruncate
```

## Testing

Because the scope of a vector bijector is very well-defined, there is a well-established testing framework to verify correctness of an implementation (`Plaice.test_all()`), which you can use in the test suite.
This function contains additional keyword arguments to control the exact testing procedure.
For example, you can test that the transformations do not cause extra allocations, should you know this to be the case for your bijector (note that this is not always possible).

For more information about generally testing bijectors (and in particular how to test Jacobians for transformations that modify the number of dimensions), see [the examples page](@ref example).

One of the most tricky parts of testing Plaice is ensuring that the transforms are compatible with automatic differentiation.
This is important for DynamicPPL: we need to be able to compute the gradient of the log-density with respect to (possibly transformed) parameters, which may include the log-abs-det-Jacobian of the transformation.
The default AD backends tested are ForwardDiff, ReverseDiff, Mooncake, and Enzyme.
It is acceptable to skip tests for a particular backend if there are genuine upstream bugs, especially with ReverseDiff, which is not actively maintained.
However where possible it is best to ensure that all backends are supported, and to use `@test_broken` to mark any known issues with specific backends.

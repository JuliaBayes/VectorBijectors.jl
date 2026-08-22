using Pkg: Pkg
Pkg.develop(; path=dirname(@__DIR__))

using Documenter
using DocumenterCodeBlocks
using DocumenterInterLinks
using Plaice
using Distributions

links = InterLinks(
    "ChangesOfVariables" => "https://juliamath.github.io/ChangesOfVariables.jl/stable/",
    "InverseFunctions" => "https://juliamath.github.io/InverseFunctions.jl/stable/",
)

makedocs(;
    sitename="Plaice",
    format=Documenter.HTML(),
    modules=[Plaice],
    pages=["index.md", "example.md"],
    checkdocs=:export,
    plugins=[CodeBlocks(), links],
)

deploydocs(; repo="github.com/JuliaBayes/Plaice.jl.git", push_preview=true)

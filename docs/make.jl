using InfiniteMPSAlgorithms
using Documenter

DocMeta.setdocmeta!(InfiniteMPSAlgorithms, :DocTestSetup,
                    :(using InfiniteMPSAlgorithms); recursive = true)

makedocs(;
    modules = [InfiniteMPSAlgorithms],
    authors = "Guo Chu <guochu604b@gmail.com>",
    sitename = "InfiniteMPSAlgorithms.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        ansicolor = true,
    ),
    pages = [
        "Home" => "index.md",
        "Conventions" => "conventions.md",
        "Algorithms" => "algorithms.md",
        "Library" => "library.md",
    ],
)

deploydocs(;
    repo = "github.com/guochu/InfiniteMPSAlgorithms.git",
    devbranch = "main",
)

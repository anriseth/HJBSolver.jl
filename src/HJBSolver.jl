isdefined(Base, :__precompile__) && __precompile__()

module HJBSolver

include("types.jl")

include("solver.jl")

include("solver2d.jl")

include("solver2d_weirdboundary.jl")

end # module

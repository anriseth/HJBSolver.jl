isdefined(Base, :__precompile__) && __precompile__()

module HJBSolver

include("types.jl")

include("solver.jl")

end # module

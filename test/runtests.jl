module HJBSolverTests
using FactCheck
using HJBSolver
using Base.Test


include("mertontest.jl")
include("boundarytest.jl")
include("test2d.jl")

FactCheck.exitstatus()
end

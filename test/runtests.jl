module HJBSolverTests
using FactCheck
using HJBSolver
using Base.Test


include("mertontest.jl")
include("boundarytest.jl")

FactCheck.exitstatus()
end

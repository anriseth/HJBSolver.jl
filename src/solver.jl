using Optim

export solve

include("policyiteration.jl") # Optimize over control in each timestep
include("policytimestep.jl")  # Choose from discrete set of controls in each timestep

function solveconstant{T<:Real}(model::HJBOneDim{T}, K::Int, N::Int, M::Int)
    # TODO: better names
    # K+1 = number of points in space domain
    # N+1 = number of points in time domain
    # M   = number of control values
    x = linspace(model.xmin, model.xmax, K+1)
    avals = linspace(model.amin, model.amax, M)
    Δx = (model.xmax-model.xmin)/K # TODO: use diff(x) to accomodate non-uniform grid
    Δτ = model.T/N # TODO: introduce non-uniform timesteps?

    vinit = zeros(x)
    for i = 1:length(x)
        vinit[i] = model.g(x[i])
    end

    v, pol = timeloopconstant(model, K, N, Δτ, vinit, avals, x, Δx)
    return v, pol
end

function solveiteration(model::HJBOneDim, K::Int, N::Int)
    # K+1 = number of points in space domain
    # N+1 = number of points in time domain
    x = linspace(model.xmin, model.xmax, K+1)
    Δx = (model.xmax-model.xmin)/K # TODO: use diff(x) to accomodate non-uniform grid
    Δτ = model.T/N # TODO: introduce non-uniform timesteps?

    vinit = zeros(x)
    for i = 1:length(x)
        vinit[i] = model.g(x[i])
    end

    v, pol = timeloopiteration(model, K, N, Δτ, vinit, x, Δx)
    return v, pol
end

# TODO: Create a better interface than this,
# maybe with a discretisation object that contains the x,a,t-arrays etc?
function solve(model::HJBOneDim, K::Int, N::Int)
    solveiteration(model, K, N)
end

function solve(model::HJBOneDim, K::Int, N::Int, M::Int)
    solveconstant(model, K, N, M)
end

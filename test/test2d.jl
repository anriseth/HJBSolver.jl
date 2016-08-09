using HJBSolver, FactCheck

type TestProblem2D{T<:Real}
    gamma::Vector{T}
    c::Vector{T}
    model::HJBTwoDim{T}
    truevaluefun::Function
    truecontrolfun::Function
end

function createmodel()
    T = 2.
    γ = [5e-2, 0.1]
    c = [-1., -0.9]

    xmin = zeros(2); xmax = 3.*ones(2)
    amin = zeros(2); amax = ones(2)

    truesol(t,x) = (t-T)*(x[1]+x[2])^2+vecdot(c,x)
    truepol(t,x) = (1+c+2*(t-T)*(x[1]+x[2]-γ.^2))./(2-2*(t-T)γ.^2)
    hatf(t,x) = (x[1]+x[2])^2+sum((1-c-2*(t-T)*(x[1]+x[2])^2)./(4-4*(t-T)*γ.^2))

    b(t,x,a) = a-1.
    σ(t,x,a) = (γ.*b(t,x,a)).^2
    f(t,x,a) = -vecdot(a,b(t,x,a)) - hatf(t,x)
    g(x) = truesol(T,x)

    model = HJBTwoDim(b, σ, f, g, T, amin, amax, xmin, xmax, truesol)
    return TestProblem2D(γ, c, model, truesol, truepol)
end

function calculateerror_iter(prob::TestProblem2D, K::Int, N::Int)
    Base.error("Implement this function")
    # TODO: calculate error at time T
    # TODO: return relative norm

    # @time v, pol = solve(prob.model, K, N)
    # model = prob.model

    # x = linspace(model.xmin, model.xmax, K+1)
    # Δτ = model.T/N

    # w = prob.truevaluefun(0., x)
    # α = prob.truecontrolfun(0., x[2:end-1])

    # return norm(w-v[:,end])/norm(w), norm(α-pol[:,end])/norm(α)
end

facts("2D, Policy iteration") do
    K = 2^8; N = 2^7
    prob = createmodel()
    # TODO: calculate error at two-three different space-time points instead
    errv, erra = calculateerror_iter(prob, K, N)

    @fact errv --> roughly(0., 1e-3)
    @fact erra --> roughly(0., 1e-2)
end

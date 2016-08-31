using HJBSolver, FactCheck

type TestProblem2D{T<:Real}
    gamma::Vector{T}
    c::Vector{T}
    model::HJBTwoDim{T}
    truevaluefun::Function
    truecontrolfun::Function
end

function createmodel(;T::Float64=1.)
    γ = [5e-2, 0.1]
    c = [-1., -0.9]

    xmin = zeros(2); xmax = [0.5,0.5]

    truesol(t,x) = (t-T)*(x[1]+x[2])^2+vecdot(c,x)
    truepol(t,x) = (1+c+2*(t-T)*(x[1]+x[2]-γ.^2))./(2-2*(t-T)*γ.^2)
    hatf(t,x) = (x[1]+x[2])^2+sum((1-c-2*(t-T)*(x[1]+x[2])).^2./(4-4*(t-T)*γ.^2))

    amin = truepol(0., xmax) - 1e-1
    amax = truepol(0., xmin) + 1e-1

    b(t,x,a) = a-1.
    σ(t,x,a) = (γ.*b(t,x,a)).^2
    f(t,x,a) = -vecdot(a,b(t,x,a)) - hatf(t,x)
    g(x) = truesol(T,x)

    model = HJBTwoDim(b, σ, f, g, T, amin, amax, xmin, xmax, truesol)
    return TestProblem2D(γ, c, model, truesol, truepol)
end


function calculateerror(prob::TestProblem2D, K::Vector{Int}, v,pol)
    model = prob.model

    x1 = linspace(model.xmin[1], model.xmax[1], K[1])
    x2 = linspace(model.xmin[2], model.xmax[2], K[2])
    x = (collect(x1), collect(x2))

    i,j = ceil(Int, K/2)
    idxi = K[2]*(i-1) + j
    xij = [x1[i], x2[j]]
    w = prob.truevaluefun(0., xij)
    α = prob.truecontrolfun(0., xij)

    return abs(w-v[idxi,1])/abs(w), abs(α[1]-pol[1][idxi,1])/abs(α[1]),
    abs(α[2]-pol[2][idxi,1])/abs(α[2])
end

facts("2D, Policy iteration") do
    K = [51, 51]; N = 20
    Δt = 2e-2
    prob = createmodel(T=Δt*N)
    model = prob.model

    @time v, pol = solve(model, K, N)
    errv, erra1, erra2 = calculateerror(prob, K, v, pol)

    @fact errv --> roughly(0., 1e-3)
    @fact erra1 --> roughly(0., 5e-2)
    @fact erra2 --> roughly(0., 5e-2)
end

facts("2D, Policy timestepping") do
    K = [51, 51]; N = 10; M = (20,20)
    Δt = 2e-2
    prob = createmodel(T=Δt*N)
    model = prob.model

    avals1 = linspace(model.amin[1], model.amax[1], M[1])
    avals2 = linspace(model.amin[2], model.amax[2], M[2])

    @time v, pol = solve(model, K, N, (avals1,avals2))

    errv, erra1, erra2 = calculateerror(prob, K, v, pol)
    @fact errv --> roughly(0., 1e-3)
    @fact erra1 --> roughly(0., 0.25)
    @fact erra2 --> roughly(0., 0.25)
end


#==

function calctruesols(t, x, prob, flat=false)
    K = [length(xi) for xi in x]
    v = zeros(reverse(K)...)
    pol1 = zeros(v)
    pol2 = zeros(v)

    for i = 1:K[1], j = 1:K[2]
        xij = [x[1][i],x[2][j]]
        v[j,i] = prob.truevaluefun(t, xij)
        pol1[j,i], pol2[j,i] = prob.truecontrolfun(t, xij)
    end
    if flat == true
        v = v[:]
        pol1 = pol1[:]
        pol2 = pol2[:]
    end
    return v, (pol1, pol2)
end

K = [51, 51]; N = 50
Δt = 2e-2
prob = createmodel(T=Δt*N)
model = prob.model
x1 = linspace(model.xmin[1], model.xmax[1], K[1])
x2 = linspace(model.xmin[2], model.xmax[2], K[2])
x = (collect(x1), collect(x2))
v, pol = solve(prob.model, K, N)

tv = zeros(v)
tpol1 = zeros(pol[1])
tpol2 = zeros(pol[2])
for i = 1:N
tv[:,i], tmppol = calctruesols((i-1)*Δt, x, prob, true)
tpol1[:,i] = tmppol[1]
tpol2[:,i] = tmppol[2]
end
tv[:,end], tmppol = calctruesols(model.T,x,prob, true)
==#

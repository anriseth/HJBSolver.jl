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
    hatf(t,x) = (x[1]+x[2])^2+sum((1-c-2*(t-T)*(x[1]+x[2])^2)./(4-4*(t-T)*γ.^2))

    amin = truepol(0., xmax) - 1e-1
    amax = truepol(0., xmin) + 1e-1

    b(t,x,a) = a-1.
    σ(t,x,a) = (γ.*b(t,x,a)).^2
    f(t,x,a) = -vecdot(a,b(t,x,a)) - hatf(t,x)
    g(x) = truesol(T,x)

    model = HJBTwoDim(b, σ, f, g, T, amin, amax, xmin, xmax, truesol)
    return TestProblem2D(γ, c, model, truesol, truepol)
end


function calculateerror_iter(prob::TestProblem2D, K::Vector{Int}, N::Int)
    @time v, pol = solve(prob.model, K, N)
    model = prob.model

    x1 = linspace(model.xmin[1], model.xmax[1], K[1])
    x2 = linspace(model.xmin[2], model.xmax[2], K[2])
    x = (collect(x1), collect(x2))

    i,j = ceil(Int, K/2)
    idxi = K[2]*(i-1) + j
    xij = [x1[i], x2[j]]
    w = prob.truevaluefun(0., xij)
    α = prob.truecontrolfun(0., xij)

    return abs(w-v[idxi,end])/abs(w), abs(α[1]-pol[1][idxi,end])/abs(α[1]),
    abs(α[2]-pol[2][idxi,end])/abs(α[2])
end

facts("2D, Policy iteration") do
    K = [51, 51]; N = 10
    prob = createmodel(T=1e-2*N)
    # TODO: calculate error at two-three different space-time points instead

    errv, erra1, erra2 = calculateerror_iter(prob, K, N)
    @fact errv --> roughly(0., 1e-10)
    @fact erra1 --> roughly(0., 1e-10)
    @fact erra2 --> roughly(0., 1e-10)

end
#     model = prob.model
#     v, pol = solve(model, K, N)
#     x1 = linspace(model.xmin[1], model.xmax[1], K[1])
#     x2 = linspace(model.xmin[2], model.xmax[2], K[2])
#     x = (collect(x1), collect(x2))

#     i,j = ceil(Int, K/2)
#     idxi = K[2]*(i-1) + j
#     xij = [x1[i], x2[j]]
#     w = prob.truevaluefun(0., xij)
#     α = prob.truecontrolfun(0., xij)


# end

function calctruesols(t, x, prob)
    K = [length(xi) for xi in x]
    v = zeros(reverse(K)...)
    pol1 = zeros(v)
    pol2 = zeros(v)

    for i = 1:K[1], j = 1:K[2]
        xij = [x[1][i],x[2][j]]
        v[j,i] = prob.truevaluefun(t, xij)
        pol1[j,i], pol2[j,i] = prob.truecontrolfun(t, xij)
    end
    return v, (pol1, pol2)
end

#facts("2D, Policy timestepping") do
K = [26, 26]; N = 4
prob = createmodel(T=1e-2*N)
model = prob.model
M = [81, 81]
avals1 = linspace(model.amin[1], model.amax[1], M[1])
avals2 = linspace(model.amin[2], model.amax[2], M[2])



# TODO: calculate error at two-three different space-time points instead
v, pol = solve(prob.model, K, N, (avals1,avals2))
#errv, erra1, erra2 = calculateerror_iter(prob, K, N)

x1 = linspace(model.xmin[1], model.xmax[1], K[1])
x2 = linspace(model.xmin[2], model.xmax[2], K[2])
x = (collect(x1), collect(x2))

i,j = ceil(Int, K/2)
idxi = K[2]*(i-1) + j
xij = [x1[i], x2[j]]
w = prob.truevaluefun(0., xij)
α = prob.truecontrolfun(0., xij)

@fact errv --> roughly(0., 1e-10)
@fact erra1 --> roughly(0., 1e-10)
@fact erra2 --> roughly(0., 1e-10)
#end


function testcoeffs{T<:Real}(I, J, V, x, Δτ::T, Δx::Vector{T})
    Base.info("Running updatecoeffs test")

    taux = Δτ ./Δx
    htaux2 = 0.5*Δτ ./ Δx.^2
    K = [length(xi) for xi in x]

    counter = 0
    # Dirichlet conditions for x_1 = {xmin, xmax}
    for i = [1, K[1]], j = 1:K[2]
        idxi = K[2]*(i-1) + j
        xij = [x[1][i], x[2][j]]
        counter = setIJV!(I,J,V,idxi,idxi,1.0,counter)
        #rhs[idxi] = model.Dbound(t, xij)
    end

    # Dirichlet conditions for x_2 = {xmin, xmax}
    for i = 2:K[1]-1, j = [1, K[2]]
        idxi = K[2]*(i-1)+j
        xij = [x[1][i], x[2][j]]
        counter = setIJV!(I,J,V,idxi,idxi,1.0,counter)
        #        rhs[idxi] = model.Dbound(t, xij)
    end

    # Interior coefficients
    for i = 2:K[1]-1, j = 2:K[2]-1
        idxi = K[2]*(i-1) + j
        idxj1f = idxi + K[2]; idxj1b = idxi - K[2]
        idxj2f = idxi + 1;    idxj2b = idxi - 1
        xij = [x[1][i], x[2][j]]
        #aij = [a1[idxi], a2[idxi]]

        #bval = model.b(t,xij,aij)
        #sval2 = model.σ(t,xij,aij).^2
        coeff1f = -(htaux2[1] + max(-1,0.)*taux[1])
        coeff1b = -(htaux2[1] - min(-1,0.)*taux[1])
        coeff2f = -(htaux2[2] + max(-1,0.)*taux[2])
        coeff2b = -(htaux2[2] - min(-1,0.)*taux[2])
        coeff0 = 1.0-(coeff1f+coeff1b + coeff2f+coeff2b)

        # TODO: does it make a performance difference what order I put these in?
        counter = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
        counter = setIJV!(I,J,V,idxi,idxj1f,coeff1f, counter)
        counter = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
        counter = setIJV!(I,J,V,idxi,idxj2f,coeff2f, counter)
        counter = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)

        #rhs[idxi] = v[idxi] + Δτ*model.f(t,xij,aij)
    end

    @assert counter == length(V)
end


function main()
    prob = createmodel()
    model = prob.model
    K = [101,101]
    N = 51
    x1 = linspace(model.xmin[1], model.xmax[1], K[1])
    x2 = linspace(model.xmin[2], model.xmax[2], K[2])
    x = (collect(x1), collect(x2))
    Δx = (model.xmax-model.xmin)./(K-1)
    Δτ = model.T/N
    @show Δx, x1[2]-x1[1], x2[2]-x2[1]

    vinit = zeros(prod(K))
    for i = 1:K[1], j = 1:K[2]
        idx = K[2]*(i-1)+j
        xij = [x[1][i], x[2][j]]
        vinit[idx] = model.g(xij)
    end
    n = length(vinit)

    # Elements in sparse system matrix (n\times n) size
    interiornnz = 5*prod(K-2)
    boundarynnz = 2*(sum(K)-2)
    totnnz = interiornnz + boundarynnz
    I = zeros(Int, totnnz); J = zeros(I); V = zeros(totnnz)
    testcoeffs(I,J,V, x, Δτ, Δx)
    Mat = sparse(I,J,V,n,n,(x,y)->Base.error("Overlap"))
    return Mat
end
#using PyPlot
#Mat = main()
#spy(Mat)

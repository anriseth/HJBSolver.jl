export solve

# TODO: better names for the functions

function setIJV!{T<:Real}(I::Vector{Int},J::Vector{Int},V::Vector{T},
                          Ival::Int,Jval::Int,Vval::T,counter::Int)
    counter = counter+1
    I[counter] = Ival
    J[counter] = Jval
    V[counter] = Vval

    return counter
end

function updatecoeffs!{T<:Real}(I, J, V, rhs, model, v, t::T, x,
                                a1::Vector{T}, a2::Vector{T}, Δτ::T, Δx::Vector{T})
    Base.info("Running updatecoeffs")
    # Updates:
    # rhs    = value function at previous timestep + f at current timestep
    # I,J,V  = vectors for creating sparse system matrix (see ?sparse)

    # Input
    # model  = HJBOneDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = tuple of vectors of x-values
    # a1     = policy-values on interior, element 1
    # a2     = policy-values on interior, element 2
    # Δτ     = time-step size
    # Δx     = spacial step length
    taux = Δτ ./Δx
    htaux2 = 0.5*Δτ ./ Δx.^2
    K = [length(xi) for xi in x]

    #TODO: add @inbounds
    counter = 0
    # Dirichlet conditions for x_1 = {xmin, xmax}
    for i = [1, K[1]], j = 1:K[2]
        idxi = K[2]*(i-1) + j
        xij = [x[1][i], x[2][j]]
        counter = setIJV!(I,J,V,idxi,idxi,1.0,counter)
        rhs[idxi] = model.Dbound(t, xij)
    end

    # Dirichlet conditions for x_2 = {xmin, xmax}
    for i = 2:K[1]-1, j = [1, K[2]]
        idxi = K[2]*(i-1)+j
        xij = [x[1][i], x[2][j]]
        counter = setIJV!(I,J,V,idxi,idxi,1.0,counter)
        rhs[idxi] = model.Dbound(t, xij)
    end

    # Interior coefficients
    for i = 2:K[1]-1, j = 2:K[2]-1
        idxi = K[2]*(i-1) + j
        idxj1f = idxi + K[2]; idxj1b = idxi - K[2]
        idxj2f = idxi + 1;    idxj2b = idxi - 1
        xij = [x[1][i], x[2][j]]
        aij = [a1[idxi], a2[idxi]]

        bval = model.b(t,xij,aij)
        sval2 = model.σ(t,xij,aij).^2
        coeff1f = -(sval2[1]*htaux2[1] + max(bval[1],0.)*taux[1])
        coeff1b = -(sval2[1]*htaux2[1] - min(bval[1],0.)*taux[1])
        coeff2f = -(sval2[2]*htaux2[2] + max(bval[2],0.)*taux[2])
        coeff2b = -(sval2[2]*htaux2[2] - min(bval[2],0.)*taux[2])
        coeff0 = 1.0-(coeff1f+coeff1b + coeff2f+coeff2b)

        # TODO: does it make a performance difference what order I put these in?
        counter = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
        counter = setIJV!(I,J,V,idxi,idxj1f,coeff1f, counter)
        counter = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
        counter = setIJV!(I,J,V,idxi,idxj2f,coeff2f, counter)
        counter = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)

        rhs[idxi] = v[idxi] + Δτ*model.f(t,xij,aij)
    end

    @assert counter == length(V)
end

function updatepol!(pol1, pol2, v, model::HJBTwoDim, t, x::Tuple, Δx::Vector;
                    tol=1e-3)
    Base.info("Running updatepol")
    # Loops over each x value and optimises the control
    # TODO: Should we instead optimize the whole control vector
    # by considering the sum of the individual objectives
    # (the gradient should be diagonal?)
    @assert size(pol1) == size(pol2)
    #TODO: add @inbounds

    K = [length(xi) for xi in x]
    invdx = 1.0 ./ Δx
    hdx2 = 0.5 ./ Δx.^2

    function hamiltonian(a, i::Int, j::Int)
        # Evaluate Hamiltonian with value a at (t,x_{i,j}), indices starting at 1
        # coeffn = values in linear system
        # e.g. coeff0  is the coefficient in front of v at x_{i,j}
        #      coeff1f is the coefficient in front of v at x_{i+1,j}
        #      coeff2b is the coefficient in front of v at x_{i,j-1}
        idxi = K[2]*(i-1) + j
        idxj1f = idxi + K[2]; idxj1b = idxi - K[2]
        idxj2f = idxi + 1;    idxj2b = idxi - 1

        xij = [x[1][i], x[2][j]]
        bval = model.b(t,xij,a)
        sval2 = model.σ(t,xij,a).^2
        coeff1f = sval2[1]*hdx2[1] + max(bval[1],0.)*invdx[1]
        coeff1b = sval2[1]*hdx2[1] - min(bval[1],0.)*invdx[1]
        coeff2f = sval2[2]*hdx2[2] + max(bval[2],0.)*invdx[2]
        coeff2b = sval2[2]*hdx2[2] - min(bval[2],0.)*invdx[2]
        return (coeff1f*(v[idxi]-v[idxj1f]) + coeff1b*(v[idxi]-v[idxj1b]) +
                coeff2f*(v[idxi]-v[idxj2f]) + coeff2b*(v[idxi]-v[idxj2b]) -
                model.f(t,xij,a))
    end

    # Only find control values at interior
    for i = 2:K[1]-1, j = 2:K[2]-1
        idxi = K[2]*(i-1) + j
        #@show [t, x[1][i], x[2][j]]
        objective(a) = hamiltonian(a, i, j)
        #g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        #diffobj = DifferentiableFunction(objective, g!)
        res = optimize(objective, [pol1[idxi], pol2[idxi]],
                       NelderMead(), OptimizationOptions(autodiff=true))#f_tol=tol,x_tol=tol,
                                                    #g_tol=tol,autodiff=true))#,
#                                                    show_trace=true,
#                                                    extended_trace=true))#,
        #model.amin, model.amax, Fminbox(),
        #optimizer = LBFGS,
        #optimizer_o = OptimizationOptions(f_tol=tol,x_tol=tol,
        #g_tol=tol))#,
        #show_trace=true))
        pol1[idxi], pol2[idxi] = res.minimum
        #@show
    end
end

function policynewtonupdate{T<:Real}(model::HJBTwoDim{T},
                                     v, a1, a2, x::Tuple{Vector{T},Vector{T}},
                                     Δx::Vector{T}, Δτ, ti::Int;
                                     tol = 1e-3,
                                     scale = 1.0,
                                     maxpolicyiter::Int = 10)
    # v  = value function at previous time-step
    # an = policy function at previous time-step / initial guess for update
    t = model.T - ti*Δτ
    @show t
    n = length(v)
    K = [length(xi) for xi in x]
    @assert length(a1) == n && length(a2) == n

    # Elements in sparse system matrix (n\times n) size
    interiornnz = 5*prod(K-2)
    boundarynnz = 2*(sum(K)-2)
    totnnz = interiornnz + boundarynnz
    I = zeros(Int, totnnz); J = zeros(I); V = zeros(T, totnnz)

    rhs = zeros(v)

    # TODO: copy or pass reference?
    pol1 = copy(a1)
    pol2 = copy(a2)
    vnew = copy(v)

    for k in 0:maxpolicyiter
        updatepol!(pol1, pol2, vnew, model, t, x, Δx)
        #@show reshape(pol1, K[2], K[1])
        updatecoeffs!(I,J,V, rhs, model, v, t, x, pol1, pol2, Δτ, Δx)

        Mat = sparse(I,J,V,n,n,(x,y)->Base.error("Overlap"))

        # TODO: Use Krylov solver for high-dimensional PDEs?
        vold = vnew
        vnew = Mat\rhs

        vchange = maximum(abs(vnew-vold)./max(1.,abs(vnew)))
        @show vchange
        if vchange < Δτ*tol && k>0
            break
        end
    end
    # TODO: do we need this one?
    updatepol!(pol1, pol2, vnew, model, t, x, Δx)

    return vnew, pol1, pol2
end

function timeloopiteration(model::HJBTwoDim, K::Vector{Int}, N::Int,
                           Δτ, vinit, x::Tuple, Δx::Vector)
    # Pass v and pol by reference?
    v = zeros(length(vinit), N+1)
    # No policy at t = T
    pol1 = zeros(length(vinit), N)
    pol2 = zeros(length(vinit), N)
    pol = (pol1, pol2)

    @inbounds v[:,1] = vinit # We use forward time t instead of backward time τ

    # initial guess for control
    pol1init = 0.5*(model.amax[1]+model.amin[1])*ones(prod(K))
    pol2init = 0.5*(model.amax[2]+model.amin[2])*ones(prod(K))
    @inbounds v[:,2], pol1[:,1], pol2[:,1] = policynewtonupdate(model, v[:,1], pol1init, pol2init,
                                                                x, Δx, Δτ, 1)
    #@show pol1[:,1]
    #@show pol2[:,1]
    #@show v[:,2]
    #Base.error("Exit")

    @inbounds for j = 2:N
        # t = (N-j)*Δτ
        # TODO: pass v-column, pol-column by reference?
        @inbounds (v[:,j+1], pol1[:,j],
                   pol2[:,j]) = policynewtonupdate(model, v[:,j], pol1[:,j-1], pol2[:,j-1],
                                                   x, Δx, Δτ, j)
    end

    return v, pol
end

function solve{T1<:Real}(model::HJBTwoDim{T1}, K::Vector{Int}, N::Int)
    # K   = number of points in each direction of the space domain
    # N+1 = number of points in time domain
    x1 = linspace(model.xmin[1], model.xmax[1], K[1])
    x2 = linspace(model.xmin[2], model.xmax[2], K[2])
    x = (collect(x1), collect(x2))
    Δx = (model.xmax-model.xmin)./(K-1) # TODO: use diff(x) to accomodate non-uniform grid
    Δτ = model.T/N # TODO: introduce non-uniform timesteps?

    vinit = zeros(T1, prod(K))
    for i = 1:K[1], j = 1:K[2]
        idx = K[2]*(i-1)+j
        xij = [x[1][i], x[2][j]]
        vinit[idx] = model.g(xij)
    end

    v, pol = timeloopiteration(model, K, N, Δτ, vinit, x, Δx)
    return v, pol
end

solve(model::HJBTwoDim, K::Int, N::Int) = solve(model, [K,K], N)

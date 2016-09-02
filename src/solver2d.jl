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

function updatesystem!{T<:Real}(I, J, V, rhs, model, v, t::T, x,
                                a1::Vector{T}, a2::Vector{T}, Δt::T, Δx::Vector{T})
    # Updates:
    # rhs    = value function at previous timestep + f at current timestep
    # I,J,V  = vectors for creating sparse system matrix (see ?sparse)

    # Input
    # model  = HJBTwoDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = tuple of vectors of x-values
    # a1     = policy-values on interior, element 1
    # a2     = policy-values on interior, element 2
    # Δt     = time-step size
    # Δx     = spacial step length
    taux = Δt ./Δx
    htaux2 = 0.5*Δt ./ Δx.^2
    K = [length(xi) for xi in x]

    #TODO: add @inbounds
    counter = 0
    # Dirichlet conditions for x_1 = {xmin, xmax}
    for i = [1, K[1]], j = 1:K[2]
        idxi = K[2]*(i-1) + j
        xij = [x[1][i], x[2][j]]
        counter = setIJV!(I,J,V,idxi,idxi,1.0,counter)
        # TODO: we could move rhs assignment outside this function
        # so it doesn't get called on every loop in the Newton solver
        rhs[idxi] = model.Dbound(t, xij)

    end

    # Dirichlet conditions for x_2 = {xmin, xmax}
    for i = 2:K[1]-1, j = [1, K[2]]
        idxi = K[2]*(i-1)+j
        xij = [x[1][i], x[2][j]]
        counter = setIJV!(I,J,V,idxi,idxi,1.0,counter)
        # TODO: we could rhs assignment outside this function
        # so it doesn't get called on every loop in the Newton solver
        rhs[idxi] = model.Dbound(t, xij)
    end

    # Interior coefficients
    for i = 2:K[1]-1, j = 2:K[2]-1
        @inbounds begin
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

            rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
        end
    end
    @assert counter == length(V)
end

function updatepol!(pol1, pol2, v, model::HJBTwoDim, t, x::Tuple, Δx::Vector;
                    tol=1e-3)
    # Loops over each x value and optimises the control
    # TODO: Should we instead optimize the whole control vector
    # by considering the sum of the individual objectives
    # (the gradient should be diagonal?)
    @assert size(pol1) == size(pol2)

    K = [length(xi) for xi in x]
    invdx = 1.0 ./ Δx
    hdx2 = 0.5 ./ Δx.^2

    function hamiltonian(a, i::Int, j::Int)
        # Evaluate Hamiltonian with value a at (t,x_{i,j}), indices starting at 1
        # coeffn = values in linear system
        # e.g. coeff0  is the coefficient in front of v at x_{i,j}
        #      coeff1f is the coefficient in front of v at x_{i+1,j}
        #      coeff2b is the coefficient in front of v at x_{i,j-1}
        @inbounds begin
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
    end

    # Only find control values at interior
    for i = 2:K[1]-1, j = 2:K[2]-1
        @inbounds begin
            idxi = K[2]*(i-1) + j
            objective(a) = hamiltonian(a, i, j)
            g!(x, out) = ForwardDiff.gradient!(out, objective, x)
            diffobj = DifferentiableFunction(objective, g!)
            res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                           model.amin, model.amax, Fminbox(),
                           optimizer = LBFGS)
            # TODO: add optimizer options?
            pol1[idxi], pol2[idxi] = res.minimum
        end
    end
end

function policynewtonupdate{T<:Real}(model::HJBTwoDim{T},
                                     v, a1, a2, x::Tuple{Vector{T},Vector{T}},
                                     Δx::Vector{T}, Δt, ti::Int;
                                     tol = 1e-3,
                                     scale = 1.0,
                                     maxpolicyiter::Int = 10)
    # v  = value function at previous time-step
    # an = policy function at previous time-step / initial guess for update
    t = (ti-1)*Δt
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
        updatesystem!(I,J,V, rhs, model, v, t, x, pol1, pol2, Δt, Δx)

        Mat = sparse(I,J,V,n,n,(x,y)->Base.error("Overlap"))

        # TODO: Use Krylov solver for high-dimensional PDEs?
        vold = vnew
        vnew = Mat\rhs

        vchange = maximum(abs(vnew-vold)./max(1.,abs(vnew)))
        if vchange < Δt*tol && k>0
            break
        end
    end
    # TODO: do we need this one?
    updatepol!(pol1, pol2, vnew, model, t, x, Δx)

    return vnew, pol1, pol2
end

function policytimestep{T<:Real}(model::HJBTwoDim{T},
                                 v, x::Tuple{Vector{T},Vector{T}},
                                 Δx::Vector{T}, Δt, ti::Int, avals::Tuple)
    # v  = value function at previous time-step
    # an = policy function at previous time-step / initial guess for update
    t = (ti-1)*Δt

    n = length(v)
    K = [length(xi) for xi in x]

    # Elements in sparse system matrix (n\times n) size
    interiornnz = 5*prod(K-2)
    boundarynnz = 2*(sum(K)-2)
    totnnz = interiornnz + boundarynnz
    I = zeros(Int, totnnz); J = zeros(I); V = zeros(T, totnnz)

    a1const = zeros(n)
    a2const = zeros(n)
    newind1 = ones(Int, n) # Indices for pol1
    newind2 = ones(Int, n) # Indices for pol2
    @inbounds vnew = -maxintfloat(T)*ones(v)
    indkeep = zeros(Bool, length(vnew)) # Indices to keep vold value

    rhs = zeros(v)

    for i = 1:length(avals[1]), j = 1:length(avals[2])
        a1const[:] = avals[1][i]
        a2const[:] = avals[2][j]

        updatesystem!(I,J,V, rhs, model, v, t, x, a1const, a2const, Δt, Δx)

        # TODO: Remove the Base.error thing, just for checking
        Mat = sparse(I,J,V,n,n,(x,y)->Base.error("Overlap"))

        # TODO: Use Krylov solver for high-dimensional PDEs?
        vold = vnew
        vnew = Mat\rhs

        indkeep[:] = vold .> vnew
        vnew[indkeep] = vold[indkeep]
        newind1[!indkeep] = i
        newind2[!indkeep] = j
    end

    pol1 = avals[1][newind1]
    pol2 = avals[2][newind2]

    return vnew, pol1, pol2
end

function timeloopiteration(model::HJBTwoDim, K::Vector{Int}, N::Int,
                           Δt, vinit, x::Tuple, Δx::Vector)
    # Pass v and pol by reference?
    v = zeros(length(vinit), N+1)
    # No policy at t = T
    pol1 = zeros(length(vinit), N)
    pol2 = zeros(length(vinit), N)
    pol = (pol1, pol2)

    @inbounds v[:,N+1] = vinit # We use forward time t instead of backward time t

    # initial guess for control
    pol1init = fill(0.5*(model.amax[1]+model.amin[1]), prod(K))
    pol2init = fill(0.5*(model.amax[2]+model.amin[2]), prod(K))
    @inbounds v[:,N], pol1[:,N], pol2[:,N] = policynewtonupdate(model, v[:,N+1], pol1init, pol2init,
                                                                x, Δx, Δt, N)

    for j = N-1:-1:1
        # t = (j-1)*Δt
        # TODO: pass v-column, pol-column by reference?
        @inbounds (v[:,j], pol1[:,j],
                   pol2[:,j]) = policynewtonupdate(model, v[:,j+1], pol1[:,j+1], pol2[:,j+1],
                                                   x, Δx, Δt, j)
    end

    return v, pol
end

function timeloopiteration(model::HJBTwoDim, K::Vector{Int}, N::Int,
                           Δt, vinit, x::Tuple, Δx::Vector, avals::Tuple)
    # Pass v and pol by reference?
    v = zeros(length(vinit), N+1)
    # No policy at t = T
    pol1 = zeros(length(vinit), N)
    pol2 = zeros(length(vinit), N)
    pol = (pol1, pol2)

    @inbounds v[:,N+1] = vinit

    @inbounds v[:,N], pol1[:,N], pol2[:,N] = policytimestep(model, v[:,N+1],
                                                            x, Δx, Δt, N, avals)

    for j = N-1:-1:1
        # t = (j-1)*Δt
        # TODO: pass v-column, pol-column by reference?
        @inbounds (v[:,j], pol1[:,j],
                   pol2[:,j]) = policytimestep(model, v[:,j+1],
                                               x, Δx, Δt, j, avals)
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
    Δt = model.T/N # TODO: introduce non-uniform timesteps?

    vinit = zeros(T1, prod(K))
    for i = 1:K[1], j = 1:K[2]
        @inbounds begin
            idx = K[2]*(i-1)+j
            xij = [x[1][i], x[2][j]]
            vinit[idx] = model.g(xij)
        end
    end

    v, pol = timeloopiteration(model, K, N, Δt, vinit, x, Δx)
    return v, pol
end

solve(model::HJBTwoDim, K::Int, N::Int) = solve(model, [K,K], N)


function solve{T1<:Real}(model::HJBTwoDim{T1}, K::Vector{Int}, N::Int,
                         avals::Tuple)
    # K   = number of points in each direction of the space domain
    # N+1 = number of points in time domain
    x1 = linspace(model.xmin[1], model.xmax[1], K[1])
    x2 = linspace(model.xmin[2], model.xmax[2], K[2])
    x = (collect(x1), collect(x2))
    Δx = (model.xmax-model.xmin)./(K-1) # TODO: use diff(x) to accomodate non-uniform grid
    Δt = model.T/N # TODO: introduce non-uniform timesteps?

    vinit = zeros(T1, prod(K))
    for i = 1:K[1], j = 1:K[2]
        @inbounds begin
            idx = K[2]*(i-1)+j
            xij = [x[1][i], x[2][j]]
            vinit[idx] = model.g(xij)
        end
    end

    v, pol = timeloopiteration(model, K, N, Δt, vinit, x, Δx, avals)
    return v, pol
end

export solve

# TODO: better names for the functions

function hamiltonianinterior(a, model::HJBTwoDim, v, t, x, xidx, Δx)
    i,j = xidx
    K = [length(xi) for xi in x]
    invdx = 1.0 ./ Δx
    hdx2 = 0.5 ./ Δx.^2

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

function setIJV!{T<:Real}(I::Vector{Int},J::Vector{Int},V::Vector{T},
                          Ival::Int,Jval::Int,Vval::T,counter::Int)
    counter = counter+1
    I[counter] = Ival
    J[counter] = Jval
    V[counter] = Vval

    return counter
end

function updateboundarysystem!{T<:Real}(I, J, V, rhs, model, v, t::T, x,
                                        a::Tuple,
                                        Δt::T, Δx::Vector{T})
    taux = Δt ./Δx
    htaux2 = 0.5*Δt ./ Δx.^2
    K = [length(xi) for xi in x]

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

    @assert counter == length(V)
end

function updateinteriorsystem!{T<:Real}(I, J, V, rhs, model, v, t::T, x,
                                        a::Tuple,
                                        Δt::T, Δx::Vector{T})
    # Updates:
    # rhs    = value function at previous timestep + f at current timestep
    # I,J,V  = vectors for creating sparse system matrix (see ?sparse)

    # Input
    # model  = HJBTwoDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = tuple of vectors of x-values
    # a      = policy-values on interior
    # Δt     = time-step size
    # Δx     = spacial step length

    taux = Δt ./Δx
    htaux2 = 0.5*Δt ./ Δx.^2
    K = [length(xi) for xi in x]

    counter = 0

    for i = 2:K[1]-1, j = 2:K[2]-1
        @inbounds begin
            idxi = K[2]*(i-1) + j
            idxj1f = idxi + K[2]; idxj1b = idxi - K[2]
            idxj2f = idxi + 1;    idxj2b = idxi - 1
            xij = [x[1][i], x[2][j]]
            aij = [ai[idxi] for ai in a]

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

function optimizepol!(pol::Tuple, objective::Function, v, model::HJBTwoDim, t,
                      x, Δx, idxi::Int;
                      tol = 1e-4, maxiter = 1000, optimizer = LBFGS,
                      verbose=false)
    initialguess = [p[idxi] for p in pol]
    # g!(a, out) = ForwardDiff.gradient!(out, objective, a)
    # diffobj = DifferentiableFunction(objective, g!)
    # res = optimize(diffobj, initialguess,
    #                model.amin, model.amax, Fminbox(),
    #                optimizer = optimizer,
    #                show_trace=verbose, extended_trace=verbose,
    #                f_tol=tol,x_tol=tol,g_tol=tol,
    #                iterations=100,
    #                optimizer_o = OptimizationOptions(f_tol=tol,x_tol=tol,
    #                                                  g_tol=tol, iterations=100,
    #                                                  show_trace=verbose,
    #                                                  extended_trace=verbose))
    res = optimize(objective, initialguess,
                   optimizer(),
                   OptimizationOptions(f_tol=tol,x_tol=tol,
                                       g_tol=tol,
                                       show_trace=verbose,
                                       extended_trace=verbose))
    for (i,val) in enumerate(Optim.minimizer(res))
        pol[i][idxi] = val
    end
end

function updateinteriorpol!(pol::Tuple, v, model::HJBTwoDim, t, x::Tuple, Δx::Vector;
                            tol = 1e-4, maxiter = 1000,
                            optimizer = LBFGS, verbose=false)
    # Loops over each x value and optimises the control
    # TODO: Should we instead optimize the whole control vector
    # by considering the sum of the individual objectives
    # (the gradient should be diagonal?)

    K = [length(xi) for xi in x]
    for i = 2:K[1]-1, j = 2:K[2]-1
        @inbounds begin
            objective(a) = hamiltonianinterior(a, model, v, t, x, [i, j], Δx)
            idxi = K[2]*(i-1) + j
            optimizepol!(pol, objective, v, model, t, x, Δx, idxi;
                         tol=tol, verbose=verbose,
                         optimizer=optimizer, maxiter=maxiter)
        end
    end
end

function policynewtonupdate{T<:Real}(model::HJBTwoDim{T},
                                     v, a::Tuple, x::Tuple{Vector{T},Vector{T}},
                                     Δx::Vector{T}, Δt, ti::Int;
                                     tol = 1e-4,
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
    #totnnz = interiornnz + boundarynnz
    #I = zeros(Int, totnnz); J = zeros(I); V = zeros(T, totnnz)
    Ii = zeros(Int, interiornnz); Ji = zeros(Ii); Vi = zeros(T, interiornnz)
    Ib = zeros(Int, boundarynnz); Jb = zeros(Ib); Vb = zeros(T, boundarynnz)

    rhs = zeros(v)

    # TODO: copy or pass reference?
    pol = (copy(a[1]), copy(a[2])) # how can we make a generator out of this?
    vnew = copy(v)

    for k in 0:maxpolicyiter
        updateinteriorpol!(pol, vnew, model, t, x, Δx)
        updateboundarysystem!(Ib,Jb,Vb, rhs, model, v, t, x, pol, Δt, Δx)
        updateinteriorsystem!(Ii,Ji,Vi, rhs, model, v, t, x, pol, Δt, Δx)

        Mat = sparse([Ib;Ii],[Jb;Ji],[Vb;Vi],n,n,(x,y)->error("Each index should be unique"))

        # TODO: Use Krylov solver for high-dimensional PDEs?
        vold = vnew
        vnew = Mat\rhs

        vchange = maximum(abs(vnew-vold)./max(1.,abs(vnew)))
        if vchange < tol && k>0
            break
        end
    end

    return vnew, pol...
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
    #totnnz = interiornnz + boundarynnz
    #I = zeros(Int, totnnz); J = zeros(I); V = zeros(T, totnnz)
    Ii = zeros(Int, interiornnz); Ji = zeros(Ii); Vi = zeros(T, interiornnz)
    Ib = zeros(Int, boundarynnz); Jb = zeros(Ib); Vb = zeros(T, boundarynnz)

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

        updateboundarysystem!(Ib,Jb,Vb, rhs, model, v, t, x, (a1const, a2const), Δt, Δx)
        updateinteriorsystem!(Ii,Ji,Vi, rhs, model, v, t, x, (a1const, a2const), Δt, Δx)

        Mat = sparse([Ib;Ii],[Jb;Ji],[Vb;Vi],n,n,(x,y)->error("Each index should be unique"))

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
    @inbounds v[:,N], pol1[:,N], pol2[:,N] = policynewtonupdate(model, v[:,N+1], (pol1init, pol2init),
                                                                x, Δx, Δt, N)

    for j = N-1:-1:1
        # t = (j-1)*Δt
        # TODO: pass v-column, pol-column by reference?
        @inbounds (v[:,j], pol1[:,j],
                   pol2[:,j]) = policynewtonupdate(model, v[:,j+1], (pol1[:,j+1], pol2[:,j+1]),
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

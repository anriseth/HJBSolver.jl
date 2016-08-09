# TODO: better names for the functions

function updatecoeffs!{T<:Real}(coeff0, coeff1f, coeff1b, coeff2f, coeff2b, rhs, model, v, t::T, x,
                                a1::Vector{T}, a1::Vector{T}, Δτ::T, Δx::Vector{T})
    # Updates:
    # rhs    = value function at previous timestep + f at current timestep
    # coeffn = values in linear system
    # e.g. coeff0  is the coefficient in front of v at x_{i,j}
    #      coeff1f is the coefficient in front of v at x_{i+1,j}
    #      coeff2b is the coefficient in front of v at x_{i,j-1}


    # Input
    # model  = HJBOneDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = vector of x-values
    # a1     = policy-values on interior, element 1
    # a2     = policy-values on interior, element 2
    # Δτ     = time-step size
    # Δx     = spacial step length
    taux = Δτ/Δx
    htaux2 = 0.5*Δτ/Δx^2

    n = length(coeff0)
    for j = 2:n-1
        bval = model.b(t,x[j],a[j-1])
        sval2 = model.σ(t,x[j],a[j-1])^2
        coeff1[j] = -(sval2*htaux2 + max(bval,0.)*taux)
        coeff2[j-1] = -(sval2*htaux2 - min(bval,0.)*taux)
        coeff0[j] = 1.-coeff1[j]-coeff2[j-1]
        rhs[j] = v[j] + Δτ*model.f(t,x[j],a[j-1])
    end
end

function updatepol!(pol1, pol2, v, model::HJBTwoDim, t, x::Tuple, Δx::Vector)
    # Loops over each x value and optimises the control
    # TODO: Should we instead optimize the whole control vector
    # by considering the sum of the individual objectives
    # (the gradient should be diagonal?)
    @assert size(pol1) == size(pol2)

    K = [length(x[1]), length(x[2])]
    idx = 1.0 ./ Δx
    hdx2 = 0.5 ./ Δx.^2

    function hamiltonian(a, i::Int, j::Int)
        # Evaluate Hamiltonian with value a at (t,x_{i,j})
        linindex = K[2]*(i-1) + j
        idx1f = linindex + K[2]; idx1b = linindex - K[2]
        idx2f = linindex + 1;    idx2b = linindex - 1

        xij = [x[1][i], x[2][j]]
        bval = model.b(t,xij,a)
        sval2 = model.σ(t,xij,a).^2
        coeff1f = sval2[1]*hdx2[1] + max(bval[1],0.)*idx[1]
        coeff1b = sval2[1]*hdx2[1] - min(bval[1],0.)*idx[1]
        coeff2f = sval2[2]*hdx2[2] + max(bval[2],0.)*idx[2]
        coeff2b = sval2[2]*hdx2[2] - min(bval[2],0.)*idx[2]
        coeff0 = -(coeff1f+coeff1b + coeff2f+coeff2b)
        return (coeff0*v[linindex] + coeff1f*v[idx1f] + coeff1b*v[idx1b] +
                coeff2f*v[idx2f] + coeff2b*v[idx2b] - model.f(t,xij,a))
    end

    # TODO: redo pol with same size as v?
    for i = 1:K[1]-2, j = 1:K[2]-2
        linidx = K[2]*(i-1) + j
        objective(a) = hamiltonian(a, i+1, j+1) # i+1,j+1 as the control only lives on the interior
        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)
        res = optimize(diffobj, [pol1[linidx], pol2[linidx]],
                       model.amin, model.amax, Fminbox(),
                       optimizer=LBFGS)

        pol1[linidx], pol2[linidx] = res.minimum
    end
end

function policynewtonupdate{T<:Real}(model::HJBTwoDim{T},
                                     v, a1, a2, x::Tuple,
                                     Δx, Δτ, ti::Int;
                                     tol = 1e-3,
                                     scale = 1.0,
                                     maxpolicyiter::Int = 10)
    # v  = value function at previous time-step
    # an = policy function at previous time-step / initial guess for update
    tol = 1e-3
    scale = 1.0
    maxpolicyiter = 10
    t = model.T - ti*Δτ
    n = length(v)
    K = [length(x[1]), length(x[2])]
    @assert length(a1) == n-2 && length(a2) == n-2

    coeff0 = ones(v)   # v_i
    coeff1 = zeros(n-1) # v_{i+1} # TODO: type stability
    coeff2 = zeros(n-1) # v_{i-1} # TODO: type stability
    rhs = zeros(v)
    # TODO: add @inbounds

    # Dirichlet conditions for x_2 = {xmin, xmax}
    for i = 1:K[1]
        # x_2 = xmin
        rhs[K[2]*(i-1)+1] = model.Dbound(t, [x[1][i], x[2][1]])
        # x_2 = xmax
        rhs[K[2]*i] = model.Dbound(t, [x[1][i], x[2][K[2]]])
    end
    # Dirichlet conditions for x_1 = {xmin, xmax}
    for i = 1:K[2]
        # x_1 = xmin
        rhs[i] = model.Dbound(t, [x[1][1], x[2][i]])
        # x_1 = xmax
        rhs[K[2]*(K[1]-1)+i] = model.Dbound(t, [x[1][K[1]], x[2][i]])
    end

    updatecoeffs!(coeff0, coeff1, coeff2, rhs, model, v, t, x, a1, a2, Δτ, Δx)
    Mat = spdiagm((coeff2, coeff0, coeff1), -1:1, n, n)
    vnew = Mat\rhs

    pol1 = copy(a1) # TODO: just update a instead?
    pol2 = copy(a2)
    updatepol!(pol1, pol2, vnew, model, t, x, Δx)

    for k in 1:maxpolicyiter
        updatecoeffs!(coeff0, coeff1, coeff2, rhs, model, v, t, x, pol, Δτ, Δx)

        Mat = spdiagm((coeff2, coeff0, coeff1), -1:1, n, n)

        # TODO: Use Krylov solver for high-dimensional PDEs
        vold = vnew
        vnew = Mat\rhs
        updatepol!(pol1, pol2, vnew, model, t, x, Δx)

        vchange = maximum(abs(vnew-vold)./max(1.,abs(vnew)))
        if vchange < tol
            break
        end
    end

    return vnew, pol1, pol2
end

function timeloopiteration(model::HJBTwoDimDim, K::Vector{Int}, N::Int,
                           Δτ, vinit, x, Δx)
    # Pass v and pol by reference?
    v = zeros(length(vinit), N+1)
    # No policy at t = T or at x-boundaries
    pol1 = zeros(prod(K-1), N)
    pol2 = zeros(prod(K-1), N)
    pol = (pol1, pol2)

    @inbounds v[:,1] = vinit # We use forward time t instead of backward time τ

    # initial guess for control
    pol1init = 0.5*(model.amax[1]-model.amin[1])*ones(prod(K-1))
    pol2init = 0.5*(model.amax[2]-model.amin[2])*ones(prod(K-1))
    @inbounds v[:,2], pol1[:,1], pol2[:,1] = policynewtonupdate(model, v[:,1], pol1init, pol2init,
                                                                x, Δx, Δτ, 1)

    @inbounds for j = 2:N
        # t = (N-j)*Δτ
        # TODO: pass v-column, pol-column by reference?
        @inbounds (v[:,j+1], pol1[:,j],
                   pol2[:,j]) = policynewtonupdate(model, v[:,j], pol1[:,j-1], pol2[:,j-1],
                                                   x, Δx, Δτ, j)
    end

    return v, pol
end

function solveiteration{T1<:Real}(model::HJBTwoDim{T1}, K::Vector{Int}, N::Int)
    # K+1 = number of points in each direction of the space domain
    # N+1 = number of points in time domain
    x1 = linspace(model.xmin[1], model.xmax[1], K[1]+1)
    x2 = linspace(model.xmin[2], model.xmax[2], K[2]+1)
    x = (collect(x1), collect(x2))
    Δx = (model.xmax-model.xmin)./K # TODO: use diff(x) to accomodate non-uniform grid
    Δτ = model.T/N # TODO: introduce non-uniform timesteps?


    vinit = zeros(T1, prod(K+1))
    for i = 0:K[1]
        offset = K[2]*i
        xij = [x[1][i+1], x[2][1]]
        for j = 1:K[2]+1
            xij[2] = x[2][j]
            vinit[offset + j] = model.g(xij)
        end
    end

    v, pol = timeloopiteration(model, K, N, Δτ, vinit, x, Δx)
    return v, pol
end

solveiteration(model::HJBTwoDim, K::Int, N::Int) = solveiteration(model, [K,K], N)

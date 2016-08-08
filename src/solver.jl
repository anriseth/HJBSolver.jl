export solve

function policytimestep(model::HJBOneDim,
                        v, avals, x, Δx, Δτ, ti::Int)
    taux = Δτ/Δx
    htaux2 = 0.5*Δτ/Δx^2
    t = model.T - ti*Δτ
    n = length(x)

    # TODO: redo this thing
    newind = ones(Int, n)
    vnew = -maxintfloat(typeof(x[1]))*ones(x)

    ind12 = zeros(Bool, length(vnew))

    coeff0 = ones(x)   # v_i
    coeff1 = zeros(n-1) # v_{i+1} # TODO: type stability
    coeff2 = zeros(n-1) # v_{i-1} # TODO: type stability
    rhs = zeros(x)
    # Dirichlet conditions
    rhs[1] = model.Dmin(t, x[1])
    rhs[end] = model.Dmin(t, x[end])

    @inbounds for i = 1:length(avals)
        a = avals[i]
        vold = vnew

        # Dirichlet conditions
        for j = 2:n-1
            bval = model.b(t,x[j],a)
            sval2 = model.σ(t,x[j],a)^2
            coeff1[j] = -(sval2*htaux2 + max(bval,0.)*taux)
            coeff2[j-1] = -(sval2*htaux2 - min(bval,0.)*taux)
            coeff0[j] = 1.-coeff1[j]-coeff2[j-1]
            rhs[j] = v[j] + Δτ*model.f(t,x[j],a)
        end

        Mat = spdiagm((coeff2, coeff0, coeff1), -1:1, n, n)

        # TODO: Use Krylov solver for high-dimensional PDEs
        vnew = Mat\rhs

        ind12[:] = vold .> vnew
        vnew[ind12] = vold[ind12]
        newind[!ind12] = i
    end
    # newind[1,end] represent boundaries, no control is used there
    pol = avals[newind[2:end-1]]

    return vnew, pol
end

function timeloop(model::HJBOneDim, K::Int, N::Int,
                  Δτ, vinit, avals, x, Δx)
    # Pass v and pol by reference?
    v = zeros(K+1, N+1)
    pol = zeros(K-1, N) # No policy at t = T or at x-boundaries

    @inbounds v[:,1] = vinit # We use forward time t instead of backward time τ

    @inbounds for j = 1:N
        # t = (N-j)*Δτ
        # TODO: pass v-column, pol-column by reference?
        v[:,j+1], pol[:,j] = policytimestep(model, v[:, j],
                                             avals, x, Δx, Δτ, j)
    end

    return v, pol
end


function solve(model::HJBOneDim, K::Int, N::Int, M::Int)
    # K+1 = number of points in space domain
    # N+1 = number of points in time domain
    # M   = number of control values
    x = linspace(model.xmin, model.xmax, K+1)
    avals = linspace(model.amin, model.amax, M)
    Δx = (model.xmax-model.xmin)/K # TODO: use diff(x) to accomodate non-uniform grid
    Δτ = model.T/N # TODO: introduce non-uniform timesteps?

    vinit::Vector{Float64} = model.g(x)

    v, pol = timeloop(model, K, N, Δτ, vinit, avals, x, Δx)
    return v, pol
end

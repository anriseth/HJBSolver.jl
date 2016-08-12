# TODO: better names for the functions
using ForwardDiff

function updatecoeffs!{T<:Real}(coeff0, coeff1, coeff2, rhs, model, v, t::T, x,
                                a::Vector{T}, Δτ::T, Δx::T)
    # Updates
    # coeffn = values in linear system
    # rhs    = value function at previous timestep + f at current timestep

    # Input
    # model  = HJBOneDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = vector of x-values
    # a      = policy-values on interior
    # Δτ     = time-step size
    # Δx     = spacial step length
    taux = Δτ/Δx
    htaux2 = 0.5*Δτ/Δx^2

    n = length(coeff0)
    for j = 2:n-1
        @inbounds begin
            bval = model.b(t,x[j],a[j-1])
            sval2 = model.σ(t,x[j],a[j-1])^2
            coeff1[j] = -(sval2*htaux2 + max(bval,0.)*taux)
            coeff2[j-1] = -(sval2*htaux2 - min(bval,0.)*taux)
            coeff0[j] = 1.-coeff1[j]-coeff2[j-1]
            rhs[j] = v[j] + Δτ*model.f(t,x[j],a[j-1])
        end
    end
end

function updatepol!(pol, v, model::HJBOneDim, t, x, Δx)
    # Loops over each x[i] value and optimises the control
    # TODO: Should we instead optimize the whole control vector
    # by considering the sum of the individual objectives
    # (the gradient should be diagonal?)

    idx = 1./Δx
    hdx2 = 0.5/Δx^2

    function hamiltonian(a, j::Int)
        @inbounds begin
            bval = model.b(t,x[j],a[1])
            sval2 = model.σ(t,x[j],a[1])^2
            coeff1 = sval2*hdx2 + max(bval,0.)*idx
            coeff2 = sval2*hdx2 - min(bval,0.)*idx
            return coeff1*(v[j]-v[j+1]) + coeff2*(v[j]-v[j-1]) - model.f(t,x[j],a[1])
        end
    end


    for j = 1:length(pol)
        objective(a) = hamiltonian(a, j+1) # j+1 as the control only lives on the interior
        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)

        # TODO: Use univariate solver until Optim/#261 is fixed
        #res = optimize(diffobj, [pol[j]], [model.amin], [model.amax],
        #               Fminbox(), optimizer=LBFGS,
        #               optimizer_o = OptimizationOptions(show_trace=true,
                                                         #extended_trace=true))
        #@inbounds pol[j] = res.minimum[1]
        # TODO: add options
        res = optimize(objective, model.amin, model.amax)
        pol[j] = res.minimum
    end
end

function policynewtonupdate{T<:Real}(model::HJBOneDim{T},
                                     v, a, x,
                                     Δx, Δτ, ti::Int;
                                     tol = 1e-3,
                                     scale = 1.0,
                                     maxpolicyiter::Int = 10)
    # v = value function at previous time-step
    # a = policy function at previous time-step / initial guess for update
    t = model.T - ti*Δτ
    n = length(x)
    @assert length(a) == n-2

    coeff0 = ones(x)   # v_i
    coeff1 = zeros(n-1) # v_{i+1} # TODO: type stability
    coeff2 = zeros(n-1) # v_{i-1} # TODO: type stability
    rhs = zeros(x)
    # Dirichlet conditions
    @inbounds rhs[1] = model.Dmin(t, x[1])
    @inbounds rhs[end] = model.Dmax(t, x[end])

    vnew = v
    pol = copy(a) # TODO: just update a instead?

    for k in 0:maxpolicyiter
        updatepol!(pol, vnew, model, t, x, Δx)
        updatecoeffs!(coeff0, coeff1, coeff2, rhs, model, v, t, x, pol, Δτ, Δx)
        Mat = spdiagm((coeff2, coeff0, coeff1), -1:1, n, n)

        # TODO: Use Krylov solver for high-dimensional PDEs
        vold = vnew
        vnew = Mat\rhs

        vchange = maximum(abs(vnew-vold)./max(scale,abs(vnew)))
        if vchange < tol && k > 0
            break
        end
    end
    updatepol!(pol, vnew, model, t, x, Δx)

    return vnew, pol
end

function timeloopiteration(model::HJBOneDim, K::Int, N::Int,
                           Δτ, vinit, x, Δx)
    # Pass v and pol by reference?
    v = zeros(K+1, N+1)
    pol = zeros(K-1, N) # No policy at t = T or at x-boundaries

    @inbounds v[:,1] = vinit # We use forward time t instead of backward time τ
    polinit = 0.5*(model.amax+model.amin)*ones(K-1) # initial guess for control
    @inbounds v[:,2], pol[:,1] = policynewtonupdate(model, v[:, 1],
                                                    polinit, x, Δx, Δτ, 1)

    for j = 2:N
        @inbounds begin
            # t = (N-j)*Δτ
            # TODO: pass v-column, pol-column by reference?
            v[:,j+1], pol[:,j] = policynewtonupdate(model, v[:, j],
                                                    pol[:,j-1], x, Δx, Δτ, j)
        end
    end

    return v, pol
end

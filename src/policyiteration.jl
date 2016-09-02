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

    let
        # Search for control on left boundary
        function objective(a)
            @inbounds begin
                bval = idx*model.b(t,x[1],a[1])
                sval2 = hdx2*model.σ(t,x[1],a[1])^2
                coeff1 = -(sval2-bval)
                coeff2 = -(bval-2*sval2)
                coeff3 = -sval2
                return coeff1*v[1] + coeff2*v[2] + coeff3*v[3] - model.f(t,x[1],a[1])
            end
        end
        res = optimize(objective, model.amin, model.amax)
        @inbounds pol[1] = res.minimum
    end

    let
        # Search for control on right boundary
        function objective(a)
            @inbounds begin
                bval = idx*model.b(t,x[end],a[1])
                sval2 = hdx2*model.σ(t,x[end],a[1])^2
                coeff1 = -(sval2+bval)
                coeff2 = bval+2*sval2
                coeff3 = -sval2
                return coeff3*v[end-2] + coeff2*v[end-1] + coeff1*v[end] - model.f(t,x[end],a[1])
            end
        end
        res = optimize(objective, model.amin, model.amax)
        @inbounds pol[end] = res.minimum
    end

    function hamiltonian(a, j::Int)
        @inbounds begin
            bval = model.b(t,x[j],a[1])
            sval2 = model.σ(t,x[j],a[1])^2
            coeff1 = sval2*hdx2 + max(bval,0.)*idx
            coeff2 = sval2*hdx2 - min(bval,0.)*idx
            return coeff1*(v[j]-v[j+1]) + coeff2*(v[j]-v[j-1]) - model.f(t,x[j],a[1])
        end
    end

    for j = 2:length(pol)-1
        objective(a) = hamiltonian(a, j)
        # g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        # diffobj = DifferentiableFunction(objective, g!)

        # res = optimize(diffobj, [pol[j]], [model.amin], [model.amax],
        #                Fminbox(), optimizer=LBFGS,
        #                optimizer_o = OptimizationOptions(g_tol=1e-3,f_tol=1e-3,x_tol=1e-3))#,
        #@inbounds pol[j] = res.minimum[1]
        # TODO: Give the choice of Brent vs Fminbox()?
        res = optimize(objective, model.amin, model.amax)
        @inbounds pol[j] = res.minimum
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
    taux = Δτ/Δx
    htaux2 = 0.5*Δτ/Δx^2

    t = (ti-1)*Δτ
    n = length(x)
    @assert length(a) == n

    coeff0 = ones(x)   # v_i
    coeff1 = zeros(n-1) # v_{i+1} # TODO: type stability
    coeff2 = zeros(n-1) # v_{i-1} # TODO: type stability
    rhs = zeros(x)
    # Dirichlet conditions
    if model.bcond[1] == true
        @inbounds rhs[1] = model.Dfun[1](t)
    end
    if model.bcond[2] == true
        @inbounds rhs[end] = model.Dfun[2](t)
    end

    vnew = v
    pol = copy(a) # TODO: just update a instead?

    for k in 0:maxpolicyiter
        updatepol!(pol, vnew, model, t, x, Δx)
        updatecoeffs!(coeff0, coeff1, coeff2, rhs, model, v, t, x, pol, Δτ, Δx)
        Mat = spdiagm((coeff2, coeff0, coeff1), -1:1, n, n)

        @inbounds begin
            # Move to sparse(I,J,K) instead?
            if model.bcond[1] == false
                bval = taux*model.b(t,x[1],pol[1])
                sval2 = htaux2*model.σ(t,x[1],pol[1])^2
                Mat[1,1:3] = [1.-(sval2-bval), -(bval-2*sval2), -sval2]
                rhs[1] = v[1] + Δτ*model.f(t,x[1],pol[1])
            end

            if model.bcond[2] == false
                bval = taux*model.b(t,x[end],pol[end])
                sval2 = htaux2*model.σ(t,x[end],pol[end])^2
                Mat[end,end-2:end] = [-sval2, bval+2*sval2, 1.-(sval2+bval)]
                rhs[end] = v[end] + Δτ*model.f(t,x[end],pol[end])
            end
        end

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

function timeloopiteration(model::HJBOneDim, N::Int,
                           Δτ, vinit, x, Δx)
    # Pass v and pol by reference?
    K = length(x)
    v = zeros(K, N+1)
    pol = zeros(K, N) # No policy at t = T

    @inbounds v[:,N+1] = vinit # We use forward time t instead of backward time τ
    polinit = fill(0.5*(model.amax+model.amin), K) # initial guess for control
    @inbounds v[:,N], pol[:,N] = policynewtonupdate(model, v[:,N+1],
                                                    polinit, x, Δx, Δτ, N)

    for j = N-1:-1:1
        @inbounds begin
            # t = (N-j)*Δτ
            # TODO: pass v-column, pol-column by reference?
            v[:,j], pol[:,j] = policynewtonupdate(model, v[:,j+1],
                                                  pol[:,j+1], x, Δx, Δτ, j)
        end
    end

    return v, pol
end

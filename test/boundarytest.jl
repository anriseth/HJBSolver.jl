using HJBSolver, FactCheck

function createmodel()
    T = 3.
    γ = 5e-2
    C = -1.

    xmin = 0.; xmax = 1.
    αmin = 0.; αmax = 1.

    d(α) = 1.-α
    b(t,x,α) = -d(α)
    σ(t,x,α) = γ*d(α)
    f(t,x,α) = α*d(α)
    g(x) = C*x
    Dmin(t) = zero(t)

    model = HJBOneDim(b, σ, f, g, T, αmin, αmax, xmin, xmax,
                      (true, false), (Dmin, null))
end


facts("1D, no boundary condition, Constant Policy") do
    K = 101; N = 60;  M = 80
    model = createmodel()

    @time v, pol = solve(model, K, N, M)

    vcheck = 0.6230938105907916
    polcheck = 0.6455696202531646

    @fact v[end,1] --> roughly(vcheck, 1e-5)
    @fact pol[end,1] --> roughly(polcheck, 1e-5)
end

facts("1D, no boundary condition, Policy-iteration") do
    K = 101; N = 60;
    model = createmodel()

    @time v, pol = solve(model, K, N)

    vcheck = 0.6246558092538466
    polcheck = 0.6398206678262762

    @fact v[end,1] --> roughly(vcheck, 1e-5)
    @fact pol[end,1] --> roughly(polcheck, 1e-5)
end


# TODO: do it for opposite boundary as well

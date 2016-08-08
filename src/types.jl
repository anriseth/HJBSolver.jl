export HJBOneDim

abstract AbstractHJB

type HJBOneDim{T1<:Real} <: AbstractHJB
    b::Function # Drift term
    σ::Function # Volatility term
    f::Function # "Gain" function
    g::Function # Terminal value
    T::T1     # Terminal time
    amin::T1  # Control minimum value
    amax::T1  # Control maximum value
    xmin::T1  # Domain minimum value
    xmax::T1  # Domain maximum value
    Dmin::Function # Dirichlet condition on x=xmin
    Dmax::Function # Dirichlet condition on x=xmax
    # TODO: assert T>0, amin < amax, xmin < xmax
end

using ITensors
using Plots
# using KrylovKit
using LinearAlgebra
using MKL

ITensors.op(::OpName"z", ::SiteType"S=1/2") = [1, 0, 0, -1]
ITensors.op(::OpName"x", ::SiteType"S=1/2") = [0, 1, 1, 0]
ITensors.op(::OpName"n", ::SiteType"S=1/2") = [0, 0, 0, 1]

mutable struct RydbergObserver <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64

    RydbergObserver(energy_tol=0.0) = new(energy_tol, 1000.0)
end

function ITensors.checkdone!(o::RydbergObserver; kwargs...)
    sw = kwargs[:sweep]
    energy = kwargs[:energy]
    if abs(energy - o.last_energy) / abs(energy) < o.energy_tol
        # early stopping
        return true
    end
    o.last_energy = energy
    return false
end

function ITensors.measure!(o::RydbergObserver; kwargs...)
    energy = kwargs[:energy]
    sweep = kwargs[:sweep]
    bond = kwargs[:bond]
    outputlevel = kwargs[:outputlevel]

    # if outputlevel > 0
    #     println("Sweep $sweep at bond $bond, the energy is $energy")
    # end
end

# assume k = 6
function blockade_radius(rabi_f)
    C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    return (C6 / rabi_f)^(1 / 6)
end

function interaction_strength(a, b)
    C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    lattice_spacing = 6 # lattice constant (micrometers)

    return C6 / (abs(a - b) * lattice_spacing)^6
end

function interactions(site)
    os = OpSum()
    for j = 1:site-1
        os += interaction_strength(site, j), "n", site, "n", j
    end
    return os
end

function rydberg(N, rabi_f, delt)
    os = OpSum()
    for site = 1:N
        os += 0.5 * rabi_f, "x", site
        os -= delt, "n", site
        os += interactions(site)
    end

    # println(combiner(sites)*contract(H))
    # @show H
    # display_mpo_elements(os)

    return os
end

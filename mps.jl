using ITensors
using Plots
# using KrylovKit
using LinearAlgebra
using MKL

ITensors.op(::OpName"z", ::SiteType"S=1/2") = [1, 0, 0, -1]
ITensors.op(::OpName"x", ::SiteType"S=1/2") = [0, 1, 1, 0]

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

    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)

    return C6 / (abs(a - b) * lattice_spacing)^6
    # if abs(a-b) == 1
    #     return V_nn
    # elseif abs(a-b) == 2
    #     return V_nnn
    # else
    #     return 0
    # end
end

function on_site_detuning(site, N)
    det = 0
    for j = 1:N
        if j != site
            det += interaction_strength(site, j)
        end
    end
    return -0.5 * det
end

function interactions(site, N)
    os = OpSum()
    for j = 1:N
        if j != site
            os += 0.125 * interaction_strength(site, j), "z", site, "z", j
        end
    end
    return os
end

function display_mpo_elements(H)
    N = length(H)
    for n = 1:N
        println("Tensor $n of the MPO:")
        println(H[n])
        println("------")
    end
end

function rydberg(N, rabi_f, delt)
    os = OpSum()
    for site = 1:N
        os += 0.5 * rabi_f, "x", site
        os += -0.5 * (delt + on_site_detuning(site, N)), "z", site
        os += interactions(site, N)
    end

    # println(combiner(sites)*contract(H))
    # @show H
    # display_mpo_elements(os)

    return os
end

function rydberg3(sites, rabi_f, delta)
    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)
    os = OpSum()
    os += 0.25 * V_nn, "Sz", 1, "Sz", 2
    os += 0.25 * V_nn, "Sz", 2, "Sz", 1
    os += 0.25 * V_nn, "Sz", 2, "Sz", 3
    os += 0.25 * V_nn, "Sz", 3, "Sz", 2
    os += 0.25 * V_nnn, "Sz", 1, "Sz", 3
    os += 0.25 * V_nnn, "Sz", 3, "Sz", 1
    os += 0.5 * rabi_f, "Sx", 1
    os += 0.5 * rabi_f, "Sx", 2
    os += 0.5 * rabi_f, "Sx", 3
    os += 0.5 * (-delta + 0.5 * (V_nn + V_nnn)), "Sz", 1
    os += 0.5 * (-delta + 0.5 * (V_nn + V_nn)), "Sz", 2
    os += 0.5 * (-delta + 0.5 * (V_nn + V_nnn)), "Sz", 3

    H = MPO(os, sites)

    return H
end

function bipartite_entropy(psi)
    b = Int(ceil(length(psi) / 2))
    # from ITensor docs
    orthogonalize!(psi, b)
    U, S, V = svd(psi[b], (linkind(psi, b - 1), siteind(psi, b)))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n, n]^2
        SvN -= p * log2(p)
    end
    return SvN
end

function mpo_to_matrix(H::MPO)
    # Convert MPO to ITensor
    T = copy(H[1])
    for n = 2:length(H)
        T *= H[n]
    end

    all_inds = inds(T)

    row_inds = all_inds[1:3]
    col_inds = all_inds[4:6]

    row_combiner = combiner(row_inds...)
    col_combiner = combiner(col_inds...)

    T_matrix = row_combiner * T * col_combiner

    return T_matrix
end

function ed_energy(N)
    sites = siteinds("S=1/2", N)

    H = MPO(rydberg(N, 2 * pi * 6.4 * (10^6), 0), sites)

    println(mpo_to_matrix(H))

    initstate(j) = isodd(j) ? "↑" : "↓"
    psi0 = randomMPS(sites, initstate; linkdims=10)

    vals, vecs, info = @time eigsolve(
        contract(H), contract(psi0), 1, :SR; ishermitian=true, tol=1e-6, krylovdim=30, eager=true
    )
    @show vals[1]
    # @show vecs
end

function ground_state(N, psi0, sites, rabi_f, delt)
    H = MPO(rydberg(N, rabi_f, delt), sites)

    nsweeps = 15
    maxdim = [10, 20, 100, 100, 200, 10, 20, 100, 100, 200, 10, 20, 100, 100, 200]
    cutoff = [1E-10]

    etol = 1E-10

    obs = RydbergObserver(etol)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, observer=obs, outputlevel=0, println=false)
    # combin = combiner(sites)
    # println(combin * contract(psi))
    # println(energy)
    entropy = bipartite_entropy(psi)
    return entropy, energy, psi
end

function avg_rydberg(psi)
    avgr = 0
    for i = 1:length(psi)
        avgr += expect(psi, [1 0; 0 0]; sites=i)
    end
    return avgr / length(psi)
end

function qpt(N, resolution, separation_assumption, start, stop, stop1, psi0, sites)
    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)
    rabi_f = 2 * pi * 6.4 * (10^6)
    C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    a = 6

    freq = (10 .^ range(start, stop=stop, length=resolution)) .* V_nn

    Y = freq
    X = collect(range(0.0, stop=stop1, length=resolution))

    energies = Array{Float64}(undef, length(Y), length(X))

    for i = 1:length(Y)
        for j = 1:length(X)
            println((i - 1) * length(X) + j)
            energies[i, j] = ground_state(N, psi0, sites, Y[i], X[j] * Y[i])[2]
        end
    end

    Y = sort(((Y .^ (-1 / 6)) .* C6^(1 / 6)) ./ a)
    energies = reverse(energies)

    second = Array{Float64}(undef, length(Y), length(X))

    # compute QPT points
    for j = 1:length(X)
        for i = 2:length(Y)-1
            # b = (energies[i,j+1]-energies[i,j]) / (X[j+1] - X[j])
            # a = (energies[i, j] - energies[i, j-1]) / (X[j] - X[j-1])
            # hessian = abs((b - a) / (X[j+1] - X[j-1])) # not hessian
            # finite_diff = (2*energies[i, j] - 5*energies[i-1, j] + 4*energies[i-2, j] - energies[i-3, j]) / (Y[i] - Y[i-1])^2
            finite_diff = energies[i+1, j] - 2 * energies[i, j] + energies[i-1, j]
            finite_diff = finite_diff / ((Y[i] - Y[i-1]) * (Y[i+1] - Y[i]))
            second[i, j] = abs(finite_diff)
        end
    end

    second = second[2:end-1, :]
    Y = Y[2:end-1]

    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    return heatmap(
        X, Y, second,
        xlims=adjusted_xlims,
        ylims=adjusted_ylims,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Second Derivative of Energy"
    )
end

function stand(N, resolution, separation_assumption, start, stop, stop2, psi0, sites)
    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)
    rabi_f = 2 * pi * 6.4 * (10^6)
    C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    a = 6

    freq = (10 .^ range(start, stop=stop, length=resolution)) .* V_nn

    Y = freq

    X = collect(range(0.0, stop=stop2, length=resolution))

    entropies = Array{Float64}(undef, length(Y), length(X))

    for i = 1:length(Y)
        for j = 1:length(X)
            println((i - 1) * length(X) + j)
            # rabi_freq = (Y[i]*a / (C6^(1/6)))^(-6)
            rabi_freq = Y[i]
            entropies[i, j] = ground_state(N, psi0, sites, rabi_freq, X[j] * rabi_freq)[1]
        end
    end

    Y = sort(((Y .^ (-1 / 6)) .* C6^(1 / 6)) ./ a)
    # Y = log2.(Y)
    entropies = reverse(entropies)

    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    return heatmap(
        X, Y, entropies,
        # xlims=adjusted_xlims,
        # ylims=adjusted_ylims,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Bipartite Entropy"
    )
end

function second(N, resolution, separation_assumption, start, stop, start1, stop1, psi0, sites)
    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)
    rabi_f = 2 * pi * 6.4 * (10^6)

    freq = (10 .^ range(start, stop=stop, length=resolution)) .* V_nn
    deltas = (10 .^ range(start1, stop=stop1, length=resolution)) .* V_nn
    entropies = Array{Float64}(undef, length(freq), length(deltas))

    # println(ground_state_entropy(N, rabi_f, 0))
    # ed_energy(N)

    # H = MPO(os, siteinds("S=1/2", 3))
    # println(contract(H))
    # sites = siteinds("S=1/2", 2)
    # println(combiner(sites) * rydberg(2, sites, rabi_f, 0))

    for i = 1:length(freq)
        for j = 1:length(deltas)
            println((i - 1) * length(deltas) + j)
            entropies[i, j] = ground_state(N, psi0, sites, freq[i], deltas[j])[1]
        end
    end

    X = log10.(freq)
    Y = log10.(deltas)

    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    return heatmap(
        X, Y, entropies,
        xlims=adjusted_xlims,
        ylims=adjusted_ylims,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="log Ω",
        ylabel="log Δ",
        colorbar_title="Bipartite Entropy"
    )
end

function frequencies(N, resolution, separation_assumption, start, stop, stop2, psi0, sites)
    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)
    rabi_f = 2 * pi * 6.4 * (10^6)
    C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    a = 6

    freq = (10 .^ range(start, stop=stop, length=resolution)) .* V_nn

    Y = freq
    X = collect(range(0.0, stop=stop2, length=resolution))

    numbers = Array{Float64}(undef, length(Y), length(X))

    for i = 1:length(Y)
        for j = 1:length(X)
            println((i - 1) * length(X) + j)
            numbers[i, j] = avg_rydberg(ground_state(N, psi0, sites, Y[i], X[j] * Y[i])[3])
        end
    end

    Y = sort(((Y .^ (-1 / 6)) .* C6^(1 / 6)) ./ a)
    numbers = reverse(numbers)
    # Y = log2.(Y)

    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    return heatmap(
        X, Y, numbers,
        # xlims=adjusted_xlims,
        # ylims=adjusted_ylims,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Average Number in Rydberg"
    )
end

function frequencies_derivative(N, resolution, separation_assumption, start, stop, stop2, psi0, sites)
    V_nn = 2 * pi * 60 * (10^6)
    V_nnn = 2 * pi * 2.3 * (10^6)
    rabi_f = 2 * pi * 6.4 * (10^6)
    C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    a = 6

    freq = (10 .^ range(start, stop=stop, length=resolution)) .* V_nn

    Y = freq
    X = collect(range(0.0, stop=stop2, length=resolution))

    numbers = Array{Float64}(undef, length(Y), length(X))

    for i = 1:length(Y)
        for j = 1:length(X)
            println((i - 1) * length(X) + j)
            numbers[i, j] = avg_rydberg(ground_state(N, psi0, sites, Y[i], X[j] * Y[i])[3])
        end
    end

    Y = sort(((Y .^ (-1 / 6)) .* C6^(1 / 6)) ./ a)
    numbers = reverse(numbers)
    # Y = log2.(Y)

    second = Array{Float64}(undef, length(Y), length(X))

    # compute QPT points
    for j = 1:length(X)
        for i = 2:length(Y)-1
            # b = (energies[i,j+1]-energies[i,j]) / (X[j+1] - X[j])
            # a = (energies[i, j] - energies[i, j-1]) / (X[j] - X[j-1])
            # hessian = abs((b - a) / (X[j+1] - X[j-1])) # not hessian
            # finite_diff = (2*numbers[i, j] - 5*numbers[i-1, j] + 4*numbers[i-2, j] - numbers[i-3, j]) / (Y[i] - Y[i-1])^2
            finite_diff = numbers[i+1, j] - 2 * numbers[i, j] + numbers[i-1, j]
            finite_diff = finite_diff / ((Y[i] - Y[i-1]) * (Y[i+1] - Y[i]))
            # finite_diff = (numbers[i+1, j] - numbers[i, j]) / (Y[i+1] - Y[i])
            # println(Y[i+1] - Y[i])
            # println(numbers[i+1, j] - numbers[i, j])
            # println()
            second[i, j] = abs(finite_diff)
        end
    end

    second = second[2:end-1, :]
    Y = Y[2:end-1]

    for i = 1:length(Y)
        for j = 2:length(X)-1
            # b = (energies[i,j+1]-energies[i,j]) / (X[j+1] - X[j])
            # a = (energies[i, j] - energies[i, j-1]) / (X[j] - X[j-1])
            # hessian = abs((b - a) / (X[j+1] - X[j-1])) # not hessian
            # finite_diff = (2*numbers[i, j] - 5*numbers[i, j-1] + 4*numbers[i, j-2] - numbers[i, j-3]) / (X[j] - X[j-1])^2
            finite_diff = numbers[i, j+1] - 2 * numbers[i, j] + numbers[i, j-1]
            finite_diff = finite_diff / (X[j] - X[j-1])^2
            second[i, j] += abs(finite_diff)

            if second[i, j] < 2
                second[i, j] = 0
            end
        end
    end

    second = second[:, 2:end-1]
    X = X[2:end-1]

    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    return heatmap(
        X, Y, second,
        # xlims=adjusted_xlims,
        # ylims=adjusted_ylims,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Hessian of Average Number in Rydberg"
    )
end

function main()
    # N = [40]
    resolution = 30
    separation_assumption = 2
    plots_matrix = []
    N = 24
    # for i = 1:length(N)
    #     println(i)
    #     sites = siteinds("S=1/2", N[i])
    #     # psi0 = MPS(sites, [i%separation_assumption == 1 ? "Up" : "Dn" for i=1:N]) # Spin down is ground state
    #     psi0 = MPS(sites, ["Up" for i=1:N[i]])
    #     push!(plots_matrix, stand(N[i], resolution, separation_assumption, -4, 3, 5, psi0, sites))
    #     push!(plots_matrix, second(N[i], resolution, separation_assumption, -4, 2, -4, 3, psi0, sites))
    #     # push!(plots_matrix, qpt(N[i], resolution, separation_assumption, -3, 3, 5))
    # end
    sites = siteinds("S=1/2", N)
    psi0 = MPS(sites, ["Dn" for i = 1:N])
    push!(plots_matrix, frequencies(N, resolution, separation_assumption, -6, -1.8, 4, psi0, sites))
    push!(plots_matrix, stand(N, resolution, separation_assumption, -6, -1.8, 4, psi0, sites))
    # sites = siteinds("S=1/2", N)
    # psi0 = MPS(sites, [i%separation_assumption == 1 ? "Up" : "Dn" for i=1:N]) # Spin down is ground state
    # push!(plots_matrix, stand(N, resolution, separation_assumption, -4.5, 3, 3.5, psi0, sites))
    # push!(plots_matrix, frequencies(N, resolution, separation_assumption, -4.5, 3, 3.5, psi0, sites))
    # sites = siteinds("S=1/2", N)
    # psi0 = MPS(sites, ["Dn" for i=1:N]) # Spin down is ground state
    # push!(plots_matrix, stand(N, resolution, separation_assumption, -4.5, 3, 3.5, psi0, sites))
    # push!(plots_matrix, frequencies(N, resolution, separation_assumption, -4.5, 3, 3.5, psi0, sites))
    # combined_ = plot(stand(N, resolution, separation_assumption, -3, 3, 5), , layout=(3, 1), size=(600, 800))
    # combined_ = plot(plots_matrix..., layout=(length(N), 2), size=(1100, length(N)*300))
    # combined_ = plot(plots_matrix..., layout=(2, 1), size=(1100, 300))
    combined_ = plot(plots_matrix..., layout=(1, 2), size=(1100, 300))
    savefig(combined_, "qpt24.png")
    return
end

main()
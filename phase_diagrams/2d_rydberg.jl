using ITensors
using Plots
using KrylovKit
using LinearAlgebra
using MKL
using ProgressMeter
using .Threads
using Measures

println(Threads.nthreads())

# ITensors.op(::OpName"z", ::SiteType"S=1/2") = [1, 0, 0, -1]
# ITensors.op(::OpName"x", ::SiteType"Qubit") = [0, 1, 1, 0]
# ITensors.op(::OpName"n", ::SiteType"S=1/2") = [0, 0, 0, 1]

# plotlyjs()
ITensors.disable_warn_order()

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
    C6 = 1 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    return (C6 / rabi_f)^(1 / 6)
end

function kagome_distance(i, j, k, l)
    if k == i
        return j - l
    end

    if (k - i) % 2 == 0
        return sqrt((j + 1 - l)^2 + (sqrt(3) * (i - k) / 2)^2)
    end

    return sqrt((j - l + 0.5)^2 + ((i - k) * sqrt(3) / 2)^2)
end

function kagome_ladder_distance(i, j, k, l)
    J1 = 1
    J2 = sqrt(2)
    J3 = 2


end

# returns the distance between two sites on a square lattice
function square_distance(i, j, k, l)
    return sqrt((j - l)^2 + (k - i)^2)
end

# returns the distance between two sites on a rhombic lattice
function rhombus_distance(i, j, k, l, theta)
    a1 = (k - i) .* [cos(theta), sin(theta)]
    a2 = (l - j) .* [1, 0]

    return norm(a1 .+ a2)
end

function interaction_strength(i, j, k, l, geometry)
    # C6 = 2 * pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    C6 = 1
    lattice_spacing = 1 # lattice constant (micrometers)
    distance = 0
    if geometry == "kagome"
        distance = kagome_distance(i, j, k, l)
    elseif geometry == "square"
        distance = square_distance(i, j, k, l)
    end

    return (C6 / distance * lattice_spacing)^6
end

function kagome_ladder_site(k, l)
    if k % 2 == 0 && (l == 1 || l == 3)
        return false
    end

    if k % 2 == 1 && l == 2
        return false
    end

    return true
end

function kagome_ladder_interactions(site, N1)
    JX = 1
    JY = 1
    JZ = sqrt(2)

    os = OpSum()
    long_neighbor = 1
    if site % 3 == 1 # axial spin
        os += JX, "ProjUp", site - 1, "ProjUp", site
        os += JY, "ProjUp", site - 2, "ProjUp", site
        os += JZ, "ProjUp", site - 3, "ProjUp", site
    elseif site % 3 == 2
        os += JX, "ProjUp", site - 1, "ProjUp", site
        os += 2 * JX, "ProjUp", site - 2, "ProjUp", site
        os += JZ, "ProjUp", site - 3, "ProjUp", site
    elseif site % 3 == 0
        os += J1 * sqrt(2), "ProjUp", site - 1, "ProjUp", site
        os += J1, "ProjUp", site - 2, "ProjUp", site
        os += J2, "ProjUp", site - 3, "ProjUp", site
    end

    for site2 = long_neighbor:site-1
        k, l = snake_site(site2, N1)
        if kagome_ladder_site(k, l) == true
            os += interaction_strength(i, j, k, l, "square"), "ProjUp", site2, "ProjUp", site
        end
    end
    return os
end

function kagome_interactions(site, i, j, N1, N2)
    os = OpSum()

    for k = 1:i
        for l = 1:j-1
            if k % 2 == 1 && l % 2 == 0
                continue
            end
            site2 = (k - 1) * N1 + l
            os += interaction_strength(i, j, k, l, "kagome"), "ProjUp", site2, "ProjUp", site
        end
    end
    return os
end

# returns i, j
function snake_site(site, N1)
    if ((site - 1) ÷ N1 + 1) % 2 == 0
        return (site - 1) ÷ N1 + 1, (site - 1) % N1 + 1
    end
    return (site - 1) ÷ N1 + 1, site % N1 + 1
end

function rhombus_interactions(site, N1, N2, theta)
    i, j = snake_site(site, N1)

    os = OpSum()
    for site2 = 1:site-1
        k, l = snake_site(site2, N1)
        os += interaction_strength(i, j, k, l, "rhombus", theta), "ProjUp", site2, "ProjUp", site
    end
    return os
end

function square_interactions(site, N1)
    os = OpSum()

    i, j = snake_site(site, N1)

    for site2 = 1:site-1
        k, l = snake_site(site2, N1)
        os += interaction_strength(i, j, k, l, "square"), "ProjUp", site2, "ProjUp", site
    end
    return os
end

function rydberg_hamiltonian(geometry, N1, N2, rabi_f, delt)
    os = OpSum()

    if geometry == "kagome"
        for i = 1:N2
            for j = 1:N1
                if i % 2 == 0 && j % 2 == 0
                    continue
                end
                site = (i - 1) * N1 + j
                os += 0.5 * rabi_f, "X", site
                os -= delt, "ProjUp", site
                os += kagome_interactions(site, i, j, N1, N2)
            end
        end
    end

    if geometry == "square"
        for site = 1:N1*N2
            os += 0.5 * rabi_f, "X", site
            os -= delt, "ProjUp", site
            os += square_interactions(site, N1)
        end
    end

    # println(combiner(sites)*contract(H))
    # @show H
    # display_mpo_elements(os)

    # println(os)

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

function bipartite_entropy(psi)
    b = Int(ceil(length(psi) / 2))
    # from ITensor docs
    ITensors.orthogonalize!(psi, b)
    U, S, V = svd(psi[b], (linkind(psi, b - 1), siteind(psi, b)))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n, n]^2
        SvN -= p * log2(p)
    end
    return SvN
end

function ground_state(geometry, N1, N2, psi0, sites, rabi_f, delt, ed_, krylov_dim=3)
    H = MPO(rydberg_hamiltonian(geometry, N1, N2, rabi_f, delt), sites)

    nsweeps = 15
    # maxdim = [100, 100, 200, 200, 400]
    maxdim = [400, 800, 1600, 2000]
    # maxdim = fill(100,nsweeps)
    cutoff = 1E-10

    etol = 1E-9

    noise = 1E-11

    obs = RydbergObserver(etol)
    if ed_ == true
        energy, psi = ed(H, psi0, sites, krylov_dim)
    end
    if ed_ == false
        energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, noise=noise, outputlevel=0, println=false)
    end
    # eigen(H, )
    # H2 = inner(H,psi,H,psi)
    # E = inner(psi',H,psi)
    # var = H2-E^2
    # println(E)
    # println(var)
    # println()

    # combin = combiner(sites)
    # println(combin * contract(psi))
    # println(energy)
    entropy = bipartite_entropy(psi)
    return entropy, energy, psi
end

function first_excitation(geometry, N1, N2, psi0, ground_psi, sites, rabi_f, delt)
    H = MPO(rydberg_hamiltonian(geometry, N1, N2, rabi_f, delt), sites)

    nsweeps = 15
    # maxdim = [100, 100, 100, 100, 100, 100, 200, 500, 1000]
    maxdim = [400, 800, 1600, 2000]
    cutoff = 1E-10
    weight = 1E6

    etol = 1E-10

    noise = 1E-11

    obs = RydbergObserver(etol)
    energy, excited_psi = dmrg(H, [ground_psi], psi0; nsweeps, maxdim, cutoff, weight=weight, noise=noise, outputlevel=0, println=false)

    # println(inner(excited_psi, ground_psi))
    return energy, excited_psi
end

function avg_rydberg(psi)
    avgr = 0
    for i = 1:length(psi)
        avgr += expect(psi, "ProjUp"; sites=i)
    end
    return avgr / length(psi)
end

function measure_mps(psi)
    measurement = []
    for i = 1:length(psi)
        push!(measurement, round(expect(psi, "ProjUp"; sites=i), digits=2))
    end
    return measurement
end

function ed(H, psi0, sites, krylov_dim)
    vals, vecs, info = @time eigsolve(
        contract(H), contract(psi0), 1, :SR; ishermitian=true, tol=1e-20, krylovdim=krylov_dim, eager=true
    )
    return vals[1], MPS(vecs[1], sites)
end

function dominant_superposition(psi)

end

function get_neighbor(arr::Array{MPS,2}, x::Int, y::Int)
    # Potential neighbors' directions
    directions = [(-1, 0), (0, -1)]

    for (dx, dy) in directions
        nx, ny = x + dx, y + dy
        if 1 <= nx <= size(arr, 1) && 1 <= ny <= size(arr, 2)
            try
                return arr[nx, ny]
            catch
                # This neighbor is uninitialized, so move on to the next one
                continue
            end
        end
    end
    return nothing
end

function focus(geometry, N1, N2, x, y, psi0, sites)
    C6 = 1
    a = 1

    rabi_freq = C6 / ((a * y)^6)
    delta = x * rabi_freq

    _, _, gs = ground_state(geometry, N1, N2, psi0, sites, rabi_freq, delta, false)

    excitation = zeros(Float64, N2, N1)

    for i = 1:N1*N2
        excitation[(i-1)÷N1+1, (i-1)%N1+1] = expect(gs, "ProjUp"; sites=i)
    end

    return heatmap(
        excitation,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="x",
        ylabel="y",
        title="R_b/a=$y, Δ/Ω=$x",
        colorbar_title="Rydberg Excitation",
        axis=false
    )
end

function stand(geometry, N1, N2, resolution, coarse, separation_assumption, start, stop, stop2, psi0, sites, ed_=true, krylov_dim=3)
    V_nn = 2 * pi * 60 * (10^6)
    V_nn = 1
    C6 = 1 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    a = 1

    freq = range(start, stop=stop, length=resolution)

    Y = freq

    X = collect(range(0.0, stop=stop2, length=resolution))

    entropies = zeros(Float64, resolution, resolution)
    numbers = zeros(Float64, resolution, resolution)
    energy_gaps = zeros(Float64, resolution, resolution)
    energies = zeros(Float64, resolution, resolution)
    second = zeros(Float64, resolution, resolution)
    hover_text = [["" for i in 1:resolution] for j in 1:resolution]

    simulation_data = Any[]

    # p = Progress(resolution^2, 1)

    ground_states = Array{MPS}(undef, resolution, resolution)
    excited_states = Array{MPS}(undef, resolution, resolution)

    p = Progress(resolution^2, 1, "Processing...", 50)
    progress = Atomic{Int64}(0)

    @threads for k = 1:resolution^2
        atomic_add!(progress, 1)
        # println(threadid())
        # if threadid() == 1
        next!(p)
        # end

        i = (k - 1) ÷ resolution + 1
        j = (k - 1) % resolution + 1

        # println((i - 1) * length(X) + j)
        # if ((i - 1) * resolution + j) % 5 == 0
        #     println((i - 1) * length(X) + j)
        # end

        rabi_freq = C6 / ((a * Y[i])^6)
        delta = X[j] * rabi_freq

        psi0_candidate = get_neighbor(ground_states, i, j)
        if psi0_candidate !== nothing
            psi0 = psi0_candidate
        end


        entropy, energy, gs = ground_state(geometry, N1, N2, psi0, sites, rabi_freq, delta, ed_, krylov_dim)

        psi0_candidate = get_neighbor(excited_states, i, j)
        if psi0_candidate !== nothing
            psi0 = psi0_candidate
        end
        energy2, s2 = first_excitation(geometry, N1, N2, psi0, gs, sites, rabi_freq, delta)

        ground_states[i, j] = gs
        excited_states[i, j] = s2

        # push!(simulation_data, (rabi_freq, delta, entropy, energy, gs))

        entropies[i, j] = entropy
        numbers[i, j] = avg_rydberg(gs)
        energy_gaps[i, j] = log(max(energy2 - energy, 1E-7))
        # energies[i, j] = energy

        mps_state = measure_mps(gs)
        hover_text[i][j] = join(string.(mps_state), ", ")
        # next!(p)
    end

    # compute QPT points
    # for j = 1:resolution
    #     for i = 2:resolution-1
    #         # b = (energies[i,j+1]-energies[i,j]) / (X[j+1] - X[j])
    #         # a = (energies[i, j] - energies[i, j-1]) / (X[j] - X[j-1])
    #         # hessian = abs((b - a) / (X[j+1] - X[j-1])) # not hessian
    #         # finite_diff = (2*energies[i, j] - 5*energies[i-1, j] + 4*energies[i-2, j] - energies[i-3, j]) / (Y[i] - Y[i-1])^2
    #         finite_diff = energies[i+1, j] - 2 * energies[i, j] + energies[i-1, j]
    #         finite_diff = finite_diff / (Y[i] - Y[i-1])^2
    #         second[i, j] = abs(finite_diff)
    #     end
    # end

    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    return simulation_data, heatmap(
        X, Y, energy_gaps,
        # xlims=adjusted_xlims,
        # ylims=adjusted_ylims,
        hover=hover_text,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Gap ln(ε)"
    ), heatmap(
        X, Y, entropies,
        # xlims=adjusted_xlims,
        # ylims=adjusted_ylims,
        hover=hover_text,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Bipartite Entropy"
    ), heatmap(
        X, Y, numbers,
        # xlims=adjusted_xlims,
        # ylims=adjusted_ylims,
        hover=hover_text,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="R_b/a",
        colorbar_title="Rydberg Fraction"
    )
end

geometry = "square"
N1 = 24 # x
N2 = 2 # y
resolution = 30
coarse_resolution = 4
separation_assumption = 2
sites = siteinds("Qubit", N1 * N2)
psi0 = MPS(sites, ["Dn" for i = 1:N1*N2])
krylov_dim = 20
simulation_data1, plot1, plot2, plot3 = stand(geometry, N1, N2, resolution, coarse_resolution, separation_assumption, 1.0, 3.0, 4, psi0, sites, false, krylov_dim)
plot_ = plot(plot1, plot2, plot3, layout=(3, 1), size=(500, 800))
savefig(plot_, geometry * "_rydberg_ladder_" * string(N1) * "_" * string(N2) * "new.png")

# plot1 = focus(geometry, N1, N2, 4.0, 1.4, psi0, sites)
# savefig(plot1, geometry * "_rydberg_striated_ladder_" * string(N1) * ".png")
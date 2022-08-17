using PyPlot, ProgressMeter, Distributions, Random
using PyPlot: matplotlib

rc("axes.spines", top=false, right=false)

tree_info_eg = Vector{Int64}[[1, 0, 0], [1, 0, 0], [2, 1, 0], [2, 1, 0], [3, 2, 1], 
                             [3, 2, 1], [4, 2, 1], [4, 2, 1], [8, 2, 1], [9, 2, 1], [9, 2, 1]];
seg_vec_eg = Vector{Float64}[[0.0, 0.0], [1, π/2], [√2, 3π/4], [√2, π/4], [√2/2, 3π/4],
                             [√2/2, π/4], [√2/2, 3π/4], [√2/2, π/4], [√2/2, π/4], [√2/2, 3π/4], [√2/2, π/4]];

function segments_lines(tree_info, seg_vec, init_pos=nothing)
    num_segments = size(tree_info)[1]
    
    pos = zeros(num_segments, 2);
    if init_pos != nothing
        pos[1, :] = init_pos
    end
    
    for j in 1:num_segments
        pos[j, :] = pos[tree_info[j][1], :] + seg_vec[j][1] * [cos(seg_vec[j][2]), sin(seg_vec[j][2])]
    end
    
    lines = []
    for j in 1:num_segments
        x1, y1 = pos[tree_info[j][1], :]
        x2, y2 = pos[j, :]
        push!(lines, [(x1, y1), (x2, y2)])
    end
    return lines, pos
end;

lines_eg, _ = segments_lines(tree_info_eg, seg_vec_eg);

rc("font", family="Meiryo") # 日本語用フォント

figure(figsize=(6, 6), dpi=100)
ax = PyPlot.axes()
for i in 1:length(tree_info_eg)
    x, y = lines_eg[i][1]
    dx, dy = lines_eg[i][2] .- lines_eg[i][1]
    arrow(x, y, dx, dy,width=0.01,head_width=0.1,head_length=0.1,length_includes_head=true,color="k")
end
scatter(0, 0, s=100, color="k") # root

text(3, 4, L"遠心性位数 $\gamma$"*"\n (centrifugal order)", size=10, color="tab:red", ha="center", va="center")
hlines_y = [0, 0, -1, 2];
for i in 1:4
    hlines(i-1, hlines_y[i], 3, color="gray", linestyle="dashed", linewidth=2, alpha=0.5)
    text(3, (i-1)+0.2, string(i-1), size=10, color="tab:red", ha="center", va="center")
end

arrowprops=Dict("arrowstyle" => "->", "color" => "tab:blue");
annotate("根 (root)", xy=(-0.1, 0), xytext=(-1.0, 0), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("根分節\n (root segments)", xy=(0, 0.5), xytext=(-1.6, 0.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("中間分節\n (internal segments)", xy=(-0.5, 1.5), xytext=(-2.2, 1.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("末端分節\n (terminal segments)", xy=(-1.25, 2.25), xytext=(-2.8, 2.25), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("分岐点\n (branch point)", xy=(1.8, 3), xytext=(0.6, 3), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("成長円錐\n (growth cone)", xy=(1.5, 3.5), xytext=(0.3, 3.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("継続点\n (continuation point)", xy=(1.5, 2.4), xytext=(1.5, 1.5), size=10, color="tab:blue", ha="center", va="center", arrowprops=arrowprops)
annotate("一組の娘枝\n (daughter branches)", xy=(-0.75, 2.25), xytext=(-1.25, 3), size=10, color="tab:purple", ha="center", va="center", arrowprops=Dict("arrowstyle" => "->", "color" => "tab:purple"))
annotate("", xy=(-1.25, 2.25), xytext=(-1.1, 2.74), ha="center", va="center", arrowprops=Dict("arrowstyle" => "->", "color" => "tab:purple"))

xlim(-4, 3); ylim(-0.3, 4); axis("off")
ax.set_aspect("equal")
tight_layout()

function branching_prob(t, dt, γ, n, C, B∞, E, S, τ)
    return (n^(-E))*B∞*exp(-t/τ)*(exp(dt/τ) - 1)*(2^(-S*γ))/C
end;

function neurite_growth_model(tree_info_init, seg_vec_init, nt, dt, B∞, E, S, τ, μₑ, σₑ,
                              turn_rate=5, max_branch_angle=0.1π, max_turn_angle=5e-3π,
                              history_num=3)
    tree_info = copy(tree_info_init)
    seg_vec = copy(seg_vec_init)
    num_branching = 0
    
    if history_num > 1
        tree_info_history = []
        seg_vec_history = []
        history_timing = floor.(Int, collect(range(1, nt, length=history_num+1)))[2:end] 
    end
    
    @showprogress for tt in 1:nt
        t = tt*dt # Current time
        
        n, C = 0.0, 0.0
        for j in 1:size(tree_info)[1]
            if tree_info[j][3] == 1
                n += 1
                C += 2 ^(-S*tree_info[j][2])
            end
        end
        C /= n
        
        for j in 1:size(tree_info)[1]
            if tree_info[j][3] == 1
                γ = tree_info[j][2]
                p_branch = branching_prob(t, dt, γ, n, C, B∞, E, S, τ)
                if p_branch > rand()
                    # Neurite branching
                    num_branching += 1
                    branch_lens = (μₑ .+ randn(2) * σₑ)*dt　# branch length
                    branch_angles = [1, -1] .* rand(Uniform(0, max_branch_angle), 2)
                    for k in 1:2 # add two daughter branches
                        push!(tree_info, [j, γ+1, 1]) 
                        push!(seg_vec, [branch_lens[k], seg_vec[j][2]+branch_angles[k]])
                    end
                    tree_info[j][3] = 0 # reset centrifugal order of original branch
                else
                    # Neurite elongation
                    Δlen = (μₑ .+ randn() * σₑ)*dt
                    p_turn = turn_rate*Δlen
                    if p_turn > rand()
                        push!(tree_info, [j, γ, 1]) # add segment
                        seg_angle = seg_vec[j][2] + rand(Uniform(-max_turn_angle, max_turn_angle))
                        push!(seg_vec, [Δlen, seg_angle])
                        tree_info[j][3] = 0 # reset centrifugal order of original branch
                    else
                        seg_vec[j][1] += Δlen
                    end
                end
            end
        end
        if history_num > 1
            if tt in history_timing
                push!(tree_info_history, copy(tree_info))
                push!(seg_vec_history, copy(seg_vec))
            end
        end
    end
    println("Num. branching: ", num_branching)
    if history_num > 1
        return tree_info_history, seg_vec_history
    else
        return tree_info, seg_vec
    end
end;

B∞ = 13.2; E = 0.319; S = -0.205; τ = 1681541; 
μₑ = 2.14e-4; σₑ = 3.98e-4;

turn_rate = 3
max_branch_angle = 0.25π
max_turn_angle = 1e-2π

T = 18*24*60*60 # duration of growth; convert 18 days to sec
dt = 200 # sec
nt = round(Int, T/dt);

history_num = 3 # 記録する状態の数; 等間隔で記録．
init_branch_num = 2 # 初めの神経突起枝の数

# 細胞体のパラメータ
tree_info_init = [[1, 0, 0]]
seg_vec_init = [[0.0, 0.0]]

# 初期神経突起のパラメータ （神経突起が放射状に出るように設定）
Random.seed!(0)
for i in 1:init_branch_num
    push!(tree_info_init, [1, 0, 1])
    push!(seg_vec_init, [(μₑ + randn()*σₑ)*dt, (i-1)/init_branch_num*2π+1e-2*randn()])
end

@time tree_info_history, seg_vec_history = neurite_growth_model(tree_info_init, seg_vec_init, nt, dt, B∞, E, S, τ, μₑ, σₑ,
                                                                turn_rate, max_branch_angle, max_turn_angle, history_num);

maxwidth, minwidth = 1.5, 0.5 # 描画する神経突起の最大/最小の太さ
days_range = floor.(Int, collect(range(1, 18, history_num+1)))[2:end]

fig, ax = subplots(1, history_num, sharex=true, sharey=true)
for i in 1:history_num
    lines, _ = segments_lines(tree_info_history[i], seg_vec_history[i]);
    linewidths = range(maxwidth, minwidth, length=length(lines));

    line_segments = matplotlib.collections.LineCollection(lines, linewidths=linewidths, color="k")
    ax[i].add_collection(line_segments)  # dendrite
    ax[i].scatter(0, 0, s=50, color="k") # soma
    ax[i].set_xlabel(L"$x\ (\mu m)$"); ax[i].set_ylabel(L"$y\ (\mu m)$")
    ax[i].set_aspect("equal")
    ax[i].set_title(string(days_range[i])*" days")
    ax[i].label_outer()
end

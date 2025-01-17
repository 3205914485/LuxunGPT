using JSON
using Plots
using Plots.PlotMeasures
using Random

preferencesD = JSON.parsefile("preferencesElo.json")
models = sort(collect(keys(preferencesD)))

initial_elo = 1200.00
elo = Dict(model => initial_elo for model in models)

# 生成所有可能的对局
matches = []
for i in eachindex(models)
    for j in eachindex(models)
        if i != j
            for _ in 1:preferencesD[models[i]][models[j]]
                push!(matches, (models[i], models[j]))
            end
        end
    end
end

function update_elo(elo, winner, loser, k=16.0)
    expected_winner = 1.0 / (1.0 + 10.0^((elo[loser] - elo[winner]) / 400.0))
    expected_loser = 1.0 / (1.0 + 10.0^((elo[winner] - elo[loser]) / 400.0))
    
    elo[winner] += k * (1.0 - expected_winner)
    elo[loser] += k * (0.0 - expected_loser)
end

# 生成随机的对局序列
Random.seed!(123)
sequences = [shuffle(matches) for _ in 1:10000]

# 计算每个序列的Elo积分历史
elo_histories = []
for seq in sequences
    elo_copy = copy(elo)
    history = []
    for (i, j) in seq
        update_elo(elo_copy, i, j)
        push!(history, copy(elo_copy))
    end
    push!(elo_histories, history)
end

#输出分数
for model in models
    elo_values = sum([[history[model] for history in elo_histories[i]] for i in 1:10000])/10000
    println(elo_values[end], " ", model)
end

begin# 创建组合图
    l = @layout [
        grid(2,4)
        a{0.3h}
    ]
    plot_grid = plot(layout=l, size=(1200, 800), legend=true, palette=:tab10, left_margin = 20px)
    # 绘制上方 8 子图
    for i in 1:8
        for model in models
            elo_values = [history[model] for history in elo_histories[i]]
            plot!(plot_grid, subplot=i, elo_values, label=model, legend=(i==0), xlabel="Match", ylabel="Elo Rating")
        end
    end
    # 绘制下方大图
    for model in models
        elo_values = sum([[history[model] for history in elo_histories[i]] for i in 1:10000]) / 10000
        plot!(plot_grid, subplot=(9), elo_values, label=model, xlabel="Match", ylabel="Elo Rating", title="Average Elo Rating")
    end
    plot!()
end

# 初始化胜率矩阵
n = length(models)
win_rates = zeros(n, n)

# 计算胜率
for (i, model1) in enumerate(models)
    for (j, model2) in enumerate(models)
        if model1 != model2
            total_matches = preferencesM[model1][model2] + preferencesM[model2][model1]
            if total_matches > 0
                win_rates[i, j] = preferencesM[model1][model2] / total_matches
            else
                win_rates[i, j] = 0.0
            end
        else
            win_rates[i, j] = 0.5  # 0.5（自己对自己的胜率）
        end
    end
end

begin# 绘制热力图
    heatmap(
        win_rates[:,end:-1:1],
        xticks=(1:n, models),
        yticks=(n:-1:1, models),
        xrotation=45,
        yrotation=0,
        xmirror=true, 
        color=:viridis,
        colorbar_title="Win Rate",
        size=(800, 800),
        aspect_ratio=:equal,
        ylims=(0.5, 10.5),
    )
    plot!()
end
using Genie, Genie.Router, Genie.Renderer.Html, JSON

const DATA_FILE = "preferencesElo.json"
const ELO_FILE = "elo_scores.json"
MODELS = ["chatglm3-6b_org", "chatglm3-6b-10000", "qwen-2.5-0.5B-instruct_org", "qwen-2.5-0.5B-instruct-10000", "qwen-2.5-1.5B-instruct_org", "qwen-2.5-1.5B-instruct-10000", "qwen-2.5-3B-instruct_org", "qwen-2.5-3B-instruct-10000", "qwen-2.5-7B-instruct-10000", "qwen-2.5-7B-instruct_org"]

K_FACTOR = 16.0  # Elo 评分调整系数

# 加载或初始化偏好数据
function load_preferences()
    if isfile(DATA_FILE)
        return JSON.parsefile(DATA_FILE)
    else
        return Dict(model => Dict(opponent => 0 for opponent in MODELS if opponent != model) for model in MODELS)
    end
end

# 加载或初始化 Elo 评分
function load_elo_scores()
    if isfile(ELO_FILE)
        return JSON.parsefile(ELO_FILE)
    else
        return Dict(model => 1200.0 for model in MODELS)  # 初始 Elo 评分为 1200
    end
end

# 保存偏好数据
function save_preferences(preferences)
    open(DATA_FILE, "w") do f
        JSON.print(f, preferences, 4)
    end
end

# 保存 Elo 评分
function save_elo_scores(elo_scores)
    open(ELO_FILE, "w") do f
        JSON.print(f, elo_scores, 4)
    end
end

# 初始化偏好数据和 Elo 评分
preferences = load_preferences()
elo_scores = load_elo_scores()

# 计算预期结果
function expected_score(player_rating, opponent_rating)
    return 1.0 / (1.0 + 10.0^((opponent_rating - player_rating) / 400.0))
end

# 更新 Elo 评分
function update_elo_scores(winner, loser, elo_scores)
    winner_rating = elo_scores[winner]
    loser_rating = elo_scores[loser]

    # 计算预期结果
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = expected_score(loser_rating, winner_rating)

    # 更新评分
    elo_scores[winner] = winner_rating + K_FACTOR * (1.0 - expected_winner)
    elo_scores[loser] = loser_rating + K_FACTOR * (0.0 - expected_loser)
end

# 使用 Elo 评分
route("/choose", method=POST) do
    model1 = Genie.params(:model1)
    model2 = Genie.params(:model2)
    choice = Genie.params(:choice)
    # 更新偏好
    if choice == model1
        preferences[model1][model2] += 1
        update_elo_scores(model1, model2, elo_scores)  # model1 获胜
    else
        preferences[model2][model1] += 1
        update_elo_scores(model2, model1, elo_scores)  # model2 获胜
    end

    save_preferences(preferences)
    save_elo_scores(elo_scores)

    redirect("/")
end

# 在结果页面展示 Elo 评分和总比较次数（按 Elo 评分降序排列）
route("/results") do
    total_comparisons = Dict{String, Int}()
    for model in MODELS
        total = 0
        for opponent in MODELS
            if opponent != model
                total += preferences[model][opponent] + preferences[opponent][model]
            end
        end
        total_comparisons[model] = total
    end

    # 按 Elo 评分降序排序
    sorted_models = sort(collect(elo_scores), by=x->x[2], rev=true)

    # 结果页面
    html("""
    <h1>模型 Elo 评分结果（按降序排列）：</h1>
    <ul>
    $(join(["<li>$(model[1]): Elo 评分 = $(round(model[2], digits=2)), 总比较次数 = $(total_comparisons[model[1]])</li>" for model in sorted_models], "\n"))
    </ul>
    <a href="/">返回</a>
    """)
end

# 读取文件的特定一行
function read_specific_line(filename, line_number)
    lines = readlines(filename)
    if line_number <= length(lines)
        return lines[line_number]
    else
        return "" # 如果行号超出范围，返回空字符串
    end
end

# 将字符串中的 \n 替换为 <br>
function replace_newlines(text)
    return replace(text, "\\n" => "<br>")
end

# 在路由中使用替换函数
route("/") do
    # 随机选择两个不同的模型
    model1, model2 = rand(MODELS, 2)
    while model1 == model2
        model2 = rand(MODELS)
    end

    # 随机选择一个行号
    file1_lines = countlines("$model1.txt")
    file2_lines = countlines("$model2.txt")
    min_lines = min(file1_lines, file2_lines) # 取两个文件的最小行数
    line_number = rand(1:min_lines) # 随机选择一个行号

    # 读取两个模型的同一行回答
    answer1 = read_specific_line("$model1.txt", line_number)
    answer2 = read_specific_line("$model2.txt", line_number)

    # 替换换行符
    answer1_html = replace_newlines(answer1)
    answer2_html = replace_newlines(answer2)

    # 渲染页面（不展示模型名字）
    html("""
    <h1>请选择更好的回答：</h1>
    <form action="/choose" method="post">
        <input type="hidden" name="model1" value="$model1">
        <input type="hidden" name="model2" value="$model2">
        <input type="hidden" name="line_number" value="$line_number">
        <p><strong>模型A:</strong> $answer1_html</p>
        <p><strong>模型B:</strong> $answer2_html</p>
        <button type="submit" name="choice" value="$model1">选择 模型A</button>
        <button type="submit" name="choice" value="$model2">选择 模型B</button>
    </form>
    <a href="/results">查看评分结果</a>
    """)
end

# 启动服务器
Genie.up()

Genie.down()
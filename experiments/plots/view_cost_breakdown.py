import matplotlib.pyplot as plt
import numpy as np

def get_cost_data():
    agents = [8, 16, 20, 32, 64, 128, 256, 512, 1024]

    control_costs = {
        'ACS':      [133.41, 159.57, 180.06, 202.86, 276.97, 343.58, 374.09, 347.26, 371.84],
        'Heuristic':[112.45, 107.16, 100.30, 121.40, 182.04, 303.37, 386.90, 389.75, 386.13],
        'RL':       [115.82, 105.96, 101.67, 100.26, 103.48,  97.96, 100.33, 100.01, 101.65]
    }

    time_costs = {
        'ACS':      [49.54,  58.90,   65.80,  75.50,  89.79, 124.75, 135.16, 131.34, 136.47],
        'Heuristic':[45.91,  50.70,   51.57,  59.64,  77.35, 115.93, 139.91, 142.99, 140.67],
        'RL':       [47.54,  54.45,   54.51,  59.95,  66.48,  71.63,  75.48,  76.15,  83.14]
    }

    return agents, control_costs, time_costs

def plot_stacked_costs(agents, control_costs, time_costs):
    x = np.arange(len(agents))
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['ACS', 'Heuristic', 'RL']

    plt.figure(figsize=(14, 7))

    # Draw grid first so it stays behind the bars
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    for i, algo in enumerate(labels):
        control = control_costs[algo]
        time = time_costs[algo]
        bar_pos = x + offsets[i]

        # Control bar
        plt.bar(
            bar_pos, control, width=bar_width, label=f'{algo} - Control',
            color=colors[i], edgecolor='black', zorder=3, linewidth=0.5
        )
        # Time bar on top
        plt.bar(
            bar_pos, time, bottom=control, width=bar_width, label=f'{algo} - Time',
            color=colors[i], hatch='//', edgecolor='black', zorder=3, linewidth=0.5
        )

    plt.xticks(x, agents)
    plt.xlabel('Number of Agents', fontsize=12, fontweight='bold')
    plt.ylabel('Total Cost (Control + Time)', fontsize=12, fontweight='bold')
    plt.title('Stacked Bar Plot of Control and Time Costs by Algorithm', fontsize=14, fontweight='bold')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

def plot_individual_cost(agents, cost_data, cost_type='control'):
    assert cost_type in ['control', 'time'], "cost_type must be 'control' or 'time'"

    x = np.arange(len(agents))
    bar_width = 0.25
    offsets = [-bar_width, 0, bar_width]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['ACS', 'Heuristic', 'RL']
    hatches = ['', '///', '\\']  # ACS: 없음, Heuristic: 우상→좌하, RL: 좌상→우하 (넓은 느낌)

    plt.figure(figsize=(14, 6))
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    for i, algo in enumerate(labels):
        costs = cost_data[algo]
        plt.bar(
            x + offsets[i], costs, width=bar_width, label=algo,
            color=colors[i], edgecolor='black', linewidth=0.5, hatch=hatches[i], zorder=3
        )

    plt.xticks(x, agents)
    plt.xlabel('Number of Agents', fontsize=12, fontweight='bold')
    plt.ylabel(f'{cost_type.capitalize()} Cost', fontsize=12, fontweight='bold')
    plt.title(f'{cost_type.capitalize()} Cost by Algorithm and Number of Agents', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------
# 메인 실행 부분
# ------------------------
if __name__ == '__main__':
    agents, control_costs, time_costs = get_cost_data()

    # 1. 스택된 총 비용 플롯
    plot_stacked_costs(agents, control_costs, time_costs)

    # 2. Control Cost만 따로
    plot_individual_cost(agents, control_costs, cost_type='control')

    # 3. Time Cost만 따로
    plot_individual_cost(agents, time_costs, cost_type='time')

import matplotlib.pyplot as plt
import numpy as np

def get_cost_data():
    agents = [8, 16, 32, 64, 128, 256, 512, 1024]

    control_costs = {
        'ACS':      [133.41, 159.57, 202.86, 276.97, 343.58, 374.09, 347.26, 371.84],
        'Heuristic':[112.45, 107.16, 121.40, 182.04, 303.37, 386.90, 389.75, 386.13],
        'RL':       [115.82, 105.96, 100.26, 103.48,  97.96, 100.33, 100.01, 101.65]
    }

    time_costs = {
        'ACS':      [49.54,  58.90,   75.50,  89.79, 124.75, 135.16, 131.34, 136.47],
        'Heuristic':[45.91,  50.70,   59.64,  77.35, 115.93, 139.91, 142.99, 140.67],
        'RL':       [47.54,  54.45,   59.95,  66.48,  71.63,  75.48,  76.15,  83.14]
    }

    return agents, control_costs, time_costs


def plot_individual_cost(agents, cost_data, cost_type='control'):
    assert cost_type in ['Control Cost', 'Convergence Time'], "cost_type must be 'Control Cost ' or 'Convergence Time'"

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

    plt.xticks(x, agents, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Number of Agents', fontsize=16, fontweight='bold')
    y_unit = ' (s)' if cost_type == 'Convergence Time' else ''
    plt.ylabel(f'{cost_type.capitalize()}{y_unit}', fontsize=16, fontweight='bold')
    # plt.title(f'{cost_type.capitalize()} by Algorithm and Number of Agents', fontsize=14, fontweight='bold')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()


# ----------------
# MAIN
# ----------------
def main():
    agents, control_costs, time_costs = get_cost_data()

    # 1. Control Cost만 따로
    plot_individual_cost(agents, control_costs, cost_type='Control Cost')

    # 2. Time Cost만 따로
    plot_individual_cost(agents, time_costs, cost_type='Convergence Time')

    print("Plots generated successfully!")


if __name__ == '__main__':
    main()

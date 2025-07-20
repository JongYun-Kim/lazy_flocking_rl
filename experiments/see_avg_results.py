import pickle
import numpy as np
import gc


def process_results(file_path, algos):
    with open(file_path, "rb") as f:
        results = pickle.load(f)
        # seeds = pickle.load(f)
        # env_config = pickle.load(f)
        # time_taken = pickle.load(f)

    print("\n\nData loaded successfully from:", file_path)

    results_per_algo = {}

    for algo in algos:
        # Get control and episode length histories
        u_hists = results[algo]["control_hists"]
        l = results[algo]["episode_lengths"]
        num_agents = u_hists[0].shape[1]

        # Filter out invalid time steps (only within the episode length)
        control_sums = []
        for arr, length in zip(u_hists, l):
            # if length > 1000:
            #     length = 1000
            sum_per_episode = np.abs(arr[:length]).mean(axis=1).sum()
            control_sums.append(sum_per_episode)

        # Get control cost and time cost
        avg_control_cost = 1.5 * np.mean(control_sums)
        avg_time_cost = np.mean(l) * 0.1

        # Get total cost
        total_cost = avg_control_cost + avg_time_cost
        results_per_algo[algo] = {
            "total_cost": total_cost,
            "control_cost": avg_control_cost,
            "time_cost": avg_time_cost,
        }

    # 결과 출력
    for algo, values in results_per_algo.items():
        print(f"{algo} with {num_agents} agents:")
        print(f"  Total   Cost: {values['total_cost']:.2f}")
        print(f"  Control Cost: {values['control_cost']:.2f}")
        print(f"  Time    Cost: {values['time_cost']:.2f}")

def main():
    algos = ["ACS", "Heuristic", "RL"]
    file_paths = [
        "./data/cl_n8_241117_180006.pkl",
        "./data/cl_n16_241117_181635.pkl",
        # "./data/cl_n20_241118_162131.pkl",
        "./data/cl_n32_241117_183334.pkl",
        "./data/cl_n64_241117_185209.pkl",
        "./data/cl_n128_241117_200041.pkl",
        "./data/cl_n256_241118_102542.pkl",
        "./data/cl_n512_241118_034754.pkl",
        "./data/cl_n1024_241119_141716.pkl",
    ]

    for file_path in file_paths:
        process_results(file_path, algos)
        gc.collect()  # Clear memory after processing each file


if __name__ == "__main__":
    print("Running see_avg_results.py...")
    main()


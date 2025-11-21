import streamlit as st
import matplotlib.pyplot as plt

from environment import IntersectionEnv
from q_learning import QLearningAgent
from main import train_and_collect, plot_metric_inline, CONFIG

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Traffic RL Simulation", layout="wide")

st.title("🚦 Reinforcement Learning Traffic Light Simulation")

st.write("""
This app simulates and trains two reinforcement learning agents (Q-Learning and SARSA) 
on a custom 4-way traffic intersection.
""")


# =====================================================================
#                RUN SINGLE SIMULATION (Q-LEARNING)
# =====================================================================

st.header("Run Q-Learning Simulation")

# UI sliders
max_steps = st.slider("Max Steps", 10, 300, 60)
arrival_n = st.slider("Arrival Prob (North)", 0.0, 1.0, 0.4)
arrival_s = st.slider("Arrival Prob (South)", 0.0, 1.0, 0.4)
arrival_e = st.slider("Arrival Prob (East)", 0.0, 1.0, 0.3)
arrival_w = st.slider("Arrival Prob (West)", 0.0, 1.0, 0.3)
depart_rate = st.slider("Departure Rate", 1, 5, 2)

if st.button("Run Simulation"):
    env = IntersectionEnv(
        max_steps=max_steps,
        arrival_probs={"N": arrival_n, "S": arrival_s, "E": arrival_e, "W": arrival_w},
        depart_rate=depart_rate,
        animation=False
    )

    agent = QLearningAgent(**CONFIG["q_learning"])

    rewards = []
    total_reward = 0

    state, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)

        total_reward += reward
        rewards.append(reward)
        state = next_state

    st.success("Simulation Completed!")

    # Reward curve
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Step-wise Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

    # Episode metrics
    metrics = env.episode_metrics()

    st.subheader("Simulation Summary")
    st.write(f"**Total Reward:** {total_reward}")
    st.write(f"**Average Waiting Time:** {metrics['average_waiting_time']:.2f}")
    st.write(f"**Average Queue Length:** {metrics['average_queue_length']:.2f}")
    st.write(f"**Throughput (Vehicles Departed):** {metrics['throughput']:.0f}")


# =====================================================================
#         TRAIN BOTH AGENTS (Q-Learning vs SARSA)
# =====================================================================

st.header("Train Both Agents (Q-Learning vs SARSA)")

if st.button("Start Training"):
    st.info("Training... please wait.")

    avg_wait, avg_queue, throughput, total_reward = train_and_collect(CONFIG)

    st.success("Training Completed!")

    # Display training curves
    st.subheader("Average Waiting Time")
    plot_metric_inline(avg_wait, "Average Waiting Time per Episode", "Avg Wait")

    st.subheader("Average Queue Length")
    plot_metric_inline(avg_queue, "Average Queue Length per Episode", "Avg Queue")

    st.subheader("Throughput (Vehicles Departed)")
    plot_metric_inline(throughput, "Vehicles Departed per Episode", "Throughput")

    st.subheader("Total Reward")
    plot_metric_inline(total_reward, "Reward Convergence", "Total Reward")

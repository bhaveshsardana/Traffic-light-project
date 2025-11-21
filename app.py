import streamlit as st
import matplotlib.pyplot as plt
from environment import IntersectionEnv
from q_learning import QLearningAgent
from main import train_and_collect, plot_metric_inline, CONFIG

st.set_page_config(page_title="Traffic RL Simulation", layout="wide")

st.title("🚦 Reinforcement Learning Traffic Light Simulation")

st.markdown("""
This app simulates a 4-way traffic intersection controlled by a Reinforcement Learning agent (Q-Learning).
""")

# ---------------- Sidebar -------------------
st.sidebar.header("Simulation Settings")
max_steps = st.sidebar.slider("Max Steps", 10, 200, 50)
arrival_n = st.sidebar.slider("Arrival Prob (North)", 0.0, 1.0, 0.5)
arrival_s = st.sidebar.slider("Arrival Prob (South)", 0.0, 1.0, 0.5)
arrival_e = st.sidebar.slider("Arrival Prob (East)", 0.0, 1.0, 0.5)
arrival_w = st.sidebar.slider("Arrival Prob (West)", 0.0, 1.0, 0.5)
depart_rate = st.sidebar.slider("Departure Rate", 1, 5, 2)

if st.sidebar.button("Run Simulation"):
    env = IntersectionEnv(
        max_steps=max_steps, 
        arrival_probs={"N": arrival_n, "S": arrival_s, "E": arrival_e, "W": arrival_w},
        depart_rate=depart_rate,
        animation=False
    )
    agent = QLearningAgent(env)

    total_reward = 0
    states, actions, rewards = [], [], []

    for _ in range(max_steps):
        s = env.get_state()
        a = agent.choose_action(s)
        s_next, r, done, info = env.step(a)

        agent.learn(s, a, r, s_next)
        
        total_reward += r
        states.append(s)
        actions.append(a)
        rewards.append(r)

        if done:
            break

    st.success("Simulation Completed!")

    # Plot reward curve
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Step-wise Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

    # Show summary
    st.subheader("Summary")
    st.write(f"**Total Reward:** {total_reward}")
    st.write(f"**Cars Arrived:** {env.stats['arrived']}")
    st.write(f"**Cars Departed:** {env.stats['departed']}")

# ---------------- Training Section --------------------
st.header("📈 Train the Agent")

episodes = st.slider("Training Episodes", 10, 200, 50)

if st.button("Train Q-Learning Agent"):
    st.info("Training... (please wait)")

    results = train_and_collect(episodes=episodes)

    st.success("Training Completed!")

    # Plot metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cumulative Reward per Episode")
        plot_metric_inline(results["episode_rewards"], "Rewards")

    with col2:
        st.subheader("Queue Length per Episode")
        plot_metric_inline(results["avg_queue_lengths"], "Average Queue Length")

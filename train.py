import tensorflow as tf
import tf_agents as tfa

from util import splitter, evaluate

def run_experiment(
        agent: tfa.agents.TFAgent,
        train_env: tfa.environments.TFEnvironment,
        eval_env: tfa.environments.TFEnvironment,
        num_iters: int,
        step_by_episode: bool,
        init_collect_steps: int,
        collect_steps_per_iter: int,
        sample_batch_size: int,
        replay_buffer_max_len: int,
        log_interval: int,
        eval_interval: int,
        save_interval: int,
        save_path: str,
    ):

    print("Initializing...")

    policy_saver = tfa.policies.policy_saver.PolicySaver(agent.policy)

    random_policy = tfa.policies.random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(),
        train_env.action_spec(),
        observation_and_action_constraint_splitter=splitter,
    )

    replay_buffer = tfa.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_len,
    )

    init_driver = (
        tfa.drivers.dynamic_episode_driver.DynamicEpisodeDriver 
        if step_by_episode else
        tfa.drivers.dynamic_step_driver.DynamicStepDriver    
    )(train_env, random_policy, [replay_buffer.add_batch], None, init_collect_steps)

    train_driver = (
        tfa.drivers.dynamic_episode_driver.DynamicEpisodeDriver 
        if step_by_episode else
        tfa.drivers.dynamic_step_driver.DynamicStepDriver    
    )(train_env, agent.collect_policy, [replay_buffer.add_batch], None, collect_steps_per_iter)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=sample_batch_size,
        num_steps=2,
    )

    dataset_iterator = iter(dataset)

    agent.train = tfa.utils.common.function(agent.train)
    agent.train_step_counter.assign(0)


    print("Initialized. ")

    print("Collecting initial data...")
    init_driver.run()
    print("Intial data collected. ")

    print("Gathering baseline performance data...")

    print(f"Baseline performance: {evaluate(eval_env, random_policy)}")

    print("Training...")
    for _ in range(num_iters):
        train_driver.run()

        experience, _ = next(dataset_iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0 or step % eval_interval == 0:
            print(f"Step {step}:")

        if step % log_interval == 0:
            print(f"loss = {train_loss}")

        if step % eval_interval == 0:
            print(f"evaluation: {evaluate(eval_env, agent.policy)}")

        if step % save_interval == 0:
            policy_saver.save(f"{save_path}/{step}")

def main(*args):
    from env import BBTfEnv, BBThreadEnv

    train_env = BBThreadEnv(32)
    eval_env = BBTfEnv()

    q_net = tfa.networks.sequential.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(20, 20)),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(800, activation="relu"),
            tf.keras.layers.Dense(400, activation="relu"),
        ]
    )

    agent = tfa.agents.dqn.dqn_agent.DdqnAgent(
        q_network=q_net,
        action_spec=train_env.action_spec(),
        time_step_spec=train_env.time_step_spec(),
        observation_and_action_constraint_splitter=splitter,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        target_update_period=10,
    )

    agent.initialize()

    run_experiment(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        num_iters=1000000,
        step_by_episode=False,
        init_collect_steps=10000,
        collect_steps_per_iter=1,
        sample_batch_size=512,
        replay_buffer_max_len=10000,
        log_interval=10000,
        eval_interval=10000,
        save_interval=100000,
        save_path="svarog_v6_data",
    )

if __name__ == "__main__":
    main()
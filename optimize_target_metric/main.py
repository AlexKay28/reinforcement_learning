import gym

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from environment.environment import CustomEnv

from stable_baselines3 import PPO


def main():
    print("init env")
    emb_size, n_classes = 8, 3
    X, y = make_classification(
        n_samples=200,
        n_features=emb_size,
        n_informative=emb_size // 2,
        n_classes=n_classes,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    train_env = CustomEnv(X_train, y_train, emb_size=emb_size, n_classes=n_classes)
    test_env = CustomEnv(X_test, y_test, emb_size=emb_size, n_classes=n_classes)

    print("fit model")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=75_000)

    print("save classifier")
    model.save("model/mlp_classifier")

    del model  # remove to demonstrate saving and loading

    print("load classifier")
    model = PPO.load("model/mlp_classifier")

    print("test")
    rewards = 0

    obs = test_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = test_env.step(action)
    print(test_env.view())


if __name__ == "__main__":
    main()

class Config:
    def __init__(self):
        """====================================================================================================
        ## General Configuration for the Program
        ===================================================================================================="""
        # - Selection of Mode
        self.mode = ['train', 'play'][1]

        # - Set the Target Score
        self.target_score_train = 5
        self.target_score_play = 3

        # - Set the Algorithm and Policy for Player 1
        self.algorithm_1p = ['rule', 'human', 'qlearning'][0]
        self.policy_1p = None

        # - Set the Algorithm and Policy for Player 2
        self.algorithm_2p = ['rule', 'human', 'qlearning'][0]
        self.policy_2p = None

        # - Set the Game Options
        self.random_serve = True

        # - Set the Random Seed for Reproducibility
        self.seed = 100

        # Set the Train Player and Opponent for Training Mode
        self.train_algorithm = ['qlearning', 'sarsa', 'dqn', 'a2c'][0]
        self.train_side = ['1p', '2p'][0]
        self.train_rewrite = False
        self.train_opponent = 'rule'
        self.train_policy = None
        self.num_episode = 1000

        # Black & White Mode
        BNW_MODE = True
        BNW_MODE_PW = 'eklrmklasd'

        """====================================================================================================
        ## Configuration for Path
        ===================================================================================================="""
        # Path for Q-Learning Outputs
        self.path_qlearning_output = './_20_model/qlearning/outputs'
        self.path_qlearning_policy = './_20_model/qlearning/outputs/policy_trained'

        # Path for SARSA Outputs
        self.path_sarsa_output = './_20_model/sarsa/outputs'
        self.path_sarsa_policy = './_20_model/sarsa/outputs/policy_trained'

        # Path for DQN Outputs
        self.path_dqn_output = './_20_model/dqn/outputs'
        self.path_dqn_policy = './_20_model/dqn/outputs/policy_trained'

        # Path for A2C Outputs
        self.path_a2c_output = './_20_model/a2c/outputs'
        self.path_a2c_policy = './_20_model/a2c/outputs/policy_trained'

class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = 2020830
        self.snr = 40
        self.K = 8
        self.beam_num = 10
        self.M_bar = 3*self.beam_num
        self.M = 256
        self.state_size = [3,self.M_bar,self.K]
        
        self.path_num = 3
        self.action_size = self.M_bar
        self.debug_mode = False
        self.use_GPU = False
        
        self.num_episodes_to_run = 50000
        self.file_to_save_data_results = None
        self.learning_rate = 0.001
        self.H_number =1000
        self.device = 'cpu'
        self.save_model = False
        #self.standard_deviation_results = 1.0
        self.show_solution_score = False
        self.hyperparameters = {"discount_rate" : 0.8, "learning_rate" : 0.001, "buffer_size" : 16000,
                                "batch_size" : 60, "mu" : 0.0, "theta" : 0.0,
                                "sigma" : 1,"epsilon_decay_rate_denominator" : 1000, "learning_iterations" : 100,
                                "update_every_n_steps" : 160, "gradient_clipping_norm": 100, "tau" : 0.9, "alpha_prioritised_replay" : 0.6,
                                "beta_prioritised_replay" :0.4, "incremental_td_error" :0.001,"exploration_cycle_episodes_length":1000}